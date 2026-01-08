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

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    const std::string collection_name = "CPP_V2_HYBRID_SEARCH";
    const std::string field_id = "id";
    const std::string field_flag = "flag";
    const std::string field_text = "text";
    const std::string field_dense = "dense";
    const std::string field_sparse = "sparse";
    const uint32_t dimension = 128;

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({field_id, milvus::DataType::INT64, "id", true, false});
    collection_schema->AddField({field_flag, milvus::DataType::INT16, "flag"});
    collection_schema->AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR, "text").WithMaxLength(1024));
    collection_schema->AddField(
        milvus::FieldSchema(field_dense, milvus::DataType::FLOAT_VECTOR, "dense vector").WithDimension(dimension));
    collection_schema->AddField(
        milvus::FieldSchema(field_sparse, milvus::DataType::SPARSE_FLOAT_VECTOR, "sparse vector"));

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    std::vector<milvus::IndexDesc> indexes = {
        milvus::IndexDesc(field_dense, "", milvus::IndexType::DISKANN, milvus::MetricType::COSINE),
        milvus::IndexDesc(field_sparse, "", milvus::IndexType::SPARSE_INVERTED_INDEX, milvus::MetricType::IP)};
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).WithIndexes(std::move(indexes)));
    util::CheckStatus("create indexes on collection", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + collection_name, status);

    // insert some rows
    const int64_t row_count = 1000;
    milvus::EntityRows rows;
    for (auto i = 0; i < row_count; ++i) {
        milvus::EntityRow row;
        row[field_id] = i;
        row[field_flag] = i % 8 + 1;
        row[field_text] = "text_" + std::to_string(i);
        row[field_dense] = util::GenerateFloatVector(dimension);
        row[field_sparse] = util::GenerateSparseVectorInJson(50, false);
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse resp_insert;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted" << std::endl;

    {
        // verify the row count of the partition is 999 by query(count(*))
        // set to STRONG level to ensure the delete request is done by server
        auto request = milvus::QueryRequest()
                           .WithCollectionName(collection_name)
                           .AddOutputField("count(*)")
                           .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query count(*)", status);
        std::cout << "count(*) = " << response.Results().GetRowCount() << std::endl;
    }

    {
        // do hybrid search
        auto sub_req1 = milvus::SubSearchRequest()
                            .WithLimit(5)
                            .WithAnnsField(field_dense)
                            .WithFilter(field_flag + " == 5")
                            .AddFloatVector(util::GenerateFloatVector(dimension));

        auto sub_req2 = milvus::SubSearchRequest()
                            .WithLimit(15)
                            .WithAnnsField(field_sparse)
                            .WithFilter(field_flag + " in [1, 3]")
                            .AddSparseVector(util::GenerateSparseVector(50));

        auto reranker = std::make_shared<milvus::WeightedRerank>(std::vector<float>{0.5, 0.5});

        auto request =
            milvus::HybridSearchRequest()
                .WithCollectionName(collection_name)
                .WithLimit(10)
                .AddSubRequest(std::make_shared<milvus::SubSearchRequest>(std::move(sub_req1)))
                .AddSubRequest(std::make_shared<milvus::SubSearchRequest>(std::move(sub_req2)))
                .WithRerank(reranker)
                .AddOutputField(field_flag)
                .AddOutputField(field_text)
                // .AddOutputField(field_sparse)
                // set to BOUNDED level to accept data inconsistence within a time window(default is 5 seconds)
                .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        milvus::SearchResponse response;
        status = client->HybridSearch(request, response);
        util::CheckStatus("search", status);

        for (auto& result : response.Results().Results()) {
            std::cout << "Result of one target vector:" << std::endl;
            milvus::EntityRows output_rows;
            status = result.OutputRows(output_rows);
            util::CheckStatus("get output rows", status);
            for (const auto& row : output_rows) {
                std::cout << "\t" << row << std::endl;
            }
        }
    }

    // drop collection
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("drop collection: " + collection_name, status);

    client->Disconnect();
    return 0;
}
