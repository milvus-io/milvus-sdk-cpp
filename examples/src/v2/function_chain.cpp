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

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {
const char* const collection_name = "CPP_V2_FUNCTION_CHAIN";
const char* const field_id = "id";
const char* const field_vector = "vector";
const uint32_t dimension = 128;

void
buildCollection(milvus::MilvusClientV2Ptr& client) {
    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, false});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));

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
        row[field_vector] = util::GenerateFloatVector(dimension);
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse resp_insert;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted" << std::endl;

    // flush so the freshly inserted data is searchable before we run the function chain search
    status = client->Flush(milvus::FlushRequest().AddCollectionName(collection_name));
    util::CheckStatus("flush", status);
}

void
searchWithFunctionChain(milvus::MilvusClientV2Ptr& client, const std::vector<float>& vector, const int64_t topk) {
    // define a function chain that runs at L2_RERANK stage: round the score, sort desc and limit
    milvus::FunctionChainExpr expr("round_decimal");
    expr.AddColumnArg("$score").AddParam("decimal", 2);

    milvus::FunctionChain chain(milvus::FunctionChainStage::L2_RERANK, "round_and_sort");
    chain.Map("$score", expr).Sort("$score", true, "").Limit(topk, 0);

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithFunctionChains({chain})
                       .WithLimit(topk)
                       .WithAnnsField(field_vector)
                       .AddOutputField(field_id)
                       .AddFloatVector(vector)
                       .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

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

    // use an all-1.0 vector to search
    std::vector<float> vector;
    vector.reserve(dimension);
    for (auto i = 0; i < dimension; i++) {
        vector.push_back(1.0);
    }

    const int64_t topk = 10;
    searchWithFunctionChain(client, vector, topk);

    client->Disconnect();
    return 0;
}
