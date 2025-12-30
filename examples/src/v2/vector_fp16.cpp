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

    const std::string collection_name = "CPP_V2_FP16_VECTOR";
    const std::string field_id = "pk";
    const std::string field_vec_fp16 = "vector_fp16";
    const std::string field_vec_bf16 = "vector_bf16";
    const std::string field_text = "text";
    const uint32_t dimension = 4;

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({field_id, milvus::DataType::INT64, "id", true, false});
    collection_schema->AddField(
        milvus::FieldSchema(field_vec_fp16, milvus::DataType::FLOAT16_VECTOR).WithDimension(dimension));
    collection_schema->AddField(
        milvus::FieldSchema(field_vec_bf16, milvus::DataType::BFLOAT16_VECTOR).WithDimension(dimension));
    collection_schema->AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(100));

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector_fp16(field_vec_fp16, "", milvus::IndexType::AUTOINDEX, milvus::MetricType::COSINE);
    milvus::IndexDesc index_vector_bf16(field_vec_bf16, "", milvus::IndexType::AUTOINDEX, milvus::MetricType::COSINE);
    status = client->CreateIndex(milvus::CreateIndexRequest()
                                     .WithCollectionName(collection_name)
                                     .AddIndex(std::move(index_vector_fp16))
                                     .AddIndex(std::move(index_vector_bf16)));
    util::CheckStatus("create indexes on collection", status);

    // insert some rows
    const int64_t row_count = 100;
    milvus::EntityRows rows;
    for (auto i = 0; i < row_count; ++i) {
        milvus::EntityRow row;
        row[field_id] = i;
        row[field_text] = "hello world " + std::to_string(i);
        row[field_vec_fp16] = util::GenerateFloatVector(dimension);
        row[field_vec_bf16] = util::GenerateFloatVector(dimension);
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse resp_insert;
    milvus::EntityRows rows_copy = rows;  // the rows are used for search later, make a copy here
    status = client->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows_copy)), resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted" << std::endl;

    // load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + collection_name, status);

    // print the original vector data
    int pk_1 = 10, pk_2 = 50;
    std::cout << "Original " << field_vec_fp16 << " No." << pk_1 << ": " << rows[pk_1][field_vec_fp16] << std::endl;
    std::cout << "Original " << field_vec_bf16 << " No." << pk_1 << ": " << rows[pk_1][field_vec_bf16] << std::endl;

    std::cout << "Original " << field_vec_fp16 << " No." << pk_2 << ": " << rows[pk_2][field_vec_fp16] << std::endl;
    std::cout << "Original " << field_vec_bf16 << " No." << pk_2 << ": " << rows[pk_2][field_vec_bf16] << std::endl;

    {
        // query
        std::string expr = field_id + " in [" + std::to_string(pk_1) + "," + std::to_string(pk_2) + "]";
        auto request = milvus::QueryRequest()
                           .WithCollectionName(collection_name)
                           .WithFilter(expr)
                           .AddOutputField(field_id)
                           .AddOutputField(field_text)
                           .AddOutputField(field_vec_fp16)
                           .AddOutputField(field_vec_bf16)
                           .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        std::cout << "Query with expression: " << expr << std::endl;
        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query", status);

        milvus::EntityRows output_rows;
        status = response.Results().OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        std::cout << "Query results:" << std::endl;
        for (const auto& row : output_rows) {
            std::cout << "\t" << row << std::endl;
        }
    }

    {
        // search
        auto request =
            milvus::SearchRequest()
                .WithCollectionName(collection_name)
                .WithLimit(3)
                .WithAnnsField(field_vec_fp16)
                .AddOutputField(field_vec_fp16)
                // set to BOUNDED level to accept data inconsistence within a time window(default is 5 seconds)
                .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED)
                .AddFloat16Vector(rows[pk_1][field_vec_fp16].get<std::vector<float>>())
                .AddFloat16Vector(rows[pk_2][field_vec_fp16].get<std::vector<float>>());
        std::cout << "Searching the No." << pk_1 << " and No." << pk_2 << " vectors." << std::endl;

        milvus::SearchResponse response;
        status = client->Search(request, response);
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

    client->Disconnect();
    return 0;
}
