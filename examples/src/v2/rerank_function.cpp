// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>
#include <thread>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {
const char* const collection_name = "CPP_V2_RERANK_FUNCTION";
const char* const field_id = "id";
const char* const field_vector = "vector";
const char* const field_year = "year";
const uint32_t dimension = 128;

void
buildCollection(milvus::MilvusClientV2Ptr& client) {
    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, false});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema->AddField({field_year, milvus::DataType::INT32});

    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus(std::string("create collection: ") + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::COSINE);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus(std::string("load collection: ") + collection_name, status);

    // insert some rows
    const int64_t row_count = 1000;
    milvus::EntityRows rows;
    for (auto i = 0; i < row_count; ++i) {
        milvus::EntityRow row;
        row[field_id] = i;
        row[field_year] = i % 125 + 1900;  // year between 1900 to 2025
        row[field_vector] = util::GenerateFloatVector(dimension);
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse resp_insert;
    milvus::EntityRows rows_copy = rows;  // the rows are used for search later, make a copy here
    status = client->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows_copy)), resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted" << std::endl;

    // get row count
    milvus::QueryResponse response;
    status = client->Query(milvus::QueryRequest()
                               .WithCollectionName(collection_name)
                               .AddOutputField("count(*)")
                               .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                           response);
    util::CheckStatus("query count(*)", status);
    std::cout << "count(*) = " << response.Results().GetRowCount() << std::endl;
}

void
searchWithRerank(milvus::MilvusClientV2Ptr& client, const std::vector<float>& vector,
                 const milvus::FunctionScorePtr& function_score, int64_t topk) {
    if (function_score) {
        std::cout << "==================== Search with function score ====================" << std::endl;
        const auto& rerankers = function_score->Functions();
        for (const auto& reranker : rerankers) {
            nlohmann::json temp1 = reranker->Params();
            nlohmann::json temp2 = reranker->InputFieldNames();
            std::cout << reranker->Name() << ", params: " << temp1.dump() << ", input field names: " << temp2.dump()
                      << std::endl;
        }
    } else {
        std::cout << "==================== Search without function score ====================" << std::endl;
    }

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithRerank(function_score)
                       .WithLimit(topk)
                       .WithAnnsField(field_vector)
                       .AddOutputField(field_id)
                       .AddOutputField(field_year)
                       .AddFloatVector(vector)
                       .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResponse response;
    auto status = client->Search(request, response);
    util::CheckStatus("search", status);

    auto& result = response.Results().Results().at(0);
    milvus::EntityRows output_rows;
    status = result.OutputRows(output_rows);
    util::CheckStatus("get output rows", status);
    for (const auto& row : output_rows) {
        std::cout << "\t" << row << std::endl;
    }
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    buildCollection(client);

    // use an all-1.0 vector to search, compare the result of with/without rerankers
    std::vector<float> vector;
    vector.reserve(dimension);
    for (auto i = 0; i < dimension; i++) {
        vector.push_back(1.0);
    }

    // without reranker
    searchWithRerank(client, vector, nullptr, 10);

    // define rerankers
    // boost: https://milvus.io/docs/boost-ranker.md
    auto boost_reranker = std::make_shared<milvus::BoostRerank>("boost on year");
    boost_reranker->SetFilter("year >= 2000");  // year >= 2000 will be boosted
    boost_reranker->SetWeight(5.0);             // boosted scores will multiply with 5.0

    // gauss decay: https://milvus.io/docs/gaussian-decay.md
    auto gauss_decay = std::make_shared<milvus::DecayRerank>("gauss decay on year");
    gauss_decay->SetFunction("gauss");
    gauss_decay->AddInputFieldName(field_year);
    gauss_decay->SetOrigin(1980);
    gauss_decay->SetOffset(20);
    gauss_decay->SetScale(50);
    gauss_decay->SetDecay(0.5);

    // exponential decay: https://milvus.io/docs/exponential-decay.md
    auto exponential_decay = std::make_shared<milvus::DecayRerank>("exponential decay on year");
    exponential_decay->SetFunction("exp");
    exponential_decay->AddInputFieldName(field_year);
    exponential_decay->SetOrigin(1950);
    exponential_decay->SetOffset(20);
    exponential_decay->SetScale(50);
    exponential_decay->SetDecay(0.5);

    // linear decay: https://milvus.io/docs/linear-decay.md
    auto linear_decay = std::make_shared<milvus::DecayRerank>("linear decay on year");
    linear_decay->SetFunction("linear");
    linear_decay->AddInputFieldName(field_year);
    linear_decay->SetOrigin(1930);
    linear_decay->SetOffset(20);
    linear_decay->SetScale(50);
    linear_decay->SetDecay(0.5);

    const int64_t topk = 20;
    // boost reranker
    {
        auto function_score = std::make_shared<milvus::FunctionScore>();
        function_score->AddFunction(boost_reranker);
        searchWithRerank(client, vector, function_score, topk);
    }

    // gauss decay reranker
    {
        auto function_score = std::make_shared<milvus::FunctionScore>();
        function_score->AddFunction(gauss_decay);
        searchWithRerank(client, vector, function_score, topk);
    }

    // exponential decay reranker
    {
        auto function_score = std::make_shared<milvus::FunctionScore>();
        function_score->AddFunction(exponential_decay);
        searchWithRerank(client, vector, function_score, topk);
    }

    // linear decay reranker
    {
        auto function_score = std::make_shared<milvus::FunctionScore>();
        function_score->AddFunction(linear_decay);
        searchWithRerank(client, vector, function_score, topk);
    }

    // multi rerankers
    {
        auto function_score = std::make_shared<milvus::FunctionScore>();
        function_score->AddFunction(boost_reranker);
        function_score->AddFunction(gauss_decay);
        searchWithRerank(client, vector, function_score, topk);
    }

    client->Disconnect();
    return 0;
}
