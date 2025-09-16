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
#include "milvus/MilvusClient.h"

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    const std::string collection_name = "TEST_CPP_FP16";
    const std::string field_id = "pk";
    const std::string field_vec_fp16 = "vector_fp16";
    const std::string field_vec_bf16 = "vector_bf16";
    const std::string field_text = "text";
    const uint32_t dimension = 4;

    // collection schema, drop and create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "id", true, false});
    collection_schema.AddField(
        milvus::FieldSchema(field_vec_fp16, milvus::DataType::FLOAT16_VECTOR).WithDimension(dimension));
    collection_schema.AddField(
        milvus::FieldSchema(field_vec_bf16, milvus::DataType::BFLOAT16_VECTOR).WithDimension(dimension));
    collection_schema.AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(100));

    status = client->DropCollection(collection_name);
    status = client->CreateCollection(collection_schema);
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector_fp16(field_vec_fp16, "", milvus::IndexType::AUTOINDEX, milvus::MetricType::COSINE);
    status = client->CreateIndex(collection_name, index_vector_fp16);
    util::CheckStatus("create index on float16 vector field", status);
    milvus::IndexDesc index_vector_bf16(field_vec_bf16, "", milvus::IndexType::AUTOINDEX, milvus::MetricType::COSINE);
    status = client->CreateIndex(collection_name, index_vector_bf16);
    util::CheckStatus("create index on bfloat16 vector field", status);

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

    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", rows, dml_results);
    util::CheckStatus("insert", status);
    std::cout << dml_results.InsertCount() << " rows inserted" << std::endl;

    // load collection
    status = client->LoadCollection(collection_name);
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
        milvus::QueryArguments q_arguments{};
        q_arguments.SetCollectionName(collection_name);
        q_arguments.SetFilter(expr);
        q_arguments.AddOutputField(field_id);
        q_arguments.AddOutputField(field_text);
        q_arguments.AddOutputField(field_vec_fp16);
        q_arguments.AddOutputField(field_vec_bf16);
        q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        std::cout << "Query with expression: " << expr << std::endl;
        milvus::QueryResults query_results{};
        status = client->Query(q_arguments, query_results);
        util::CheckStatus("query", status);

        milvus::EntityRows output_rows;
        status = query_results.OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        std::cout << "Query results:" << std::endl;
        for (const auto& row : output_rows) {
            std::cout << "\t" << row << std::endl;
        }
    }

    {
        // search
        milvus::SearchArguments s_arguments{};
        s_arguments.SetCollectionName(collection_name);
        s_arguments.SetLimit(3);
        s_arguments.AddOutputField(field_vec_fp16);
        // set to BOUNDED level to accept data inconsistence within a time window(default is 5 seconds)
        s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        s_arguments.AddFloat16Vector(field_vec_fp16, rows[pk_1][field_vec_fp16].get<std::vector<float>>());
        s_arguments.AddFloat16Vector(field_vec_fp16, rows[pk_2][field_vec_fp16].get<std::vector<float>>());
        std::cout << "Searching the No." << pk_1 << " and No." << pk_2 << " vectors." << std::endl;

        milvus::SearchResults search_results{};
        status = client->Search(s_arguments, search_results);
        util::CheckStatus("search", status);

        for (auto& result : search_results.Results()) {
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
