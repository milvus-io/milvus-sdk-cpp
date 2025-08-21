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
    util::CheckStatus("Failed to connect milvus server:", status);
    std::cout << "Connect to milvus server." << std::endl;

    // drop the collection if it exists
    const std::string collection_name = "TEST_CPP_JSON";
    status = client->DropCollection(collection_name);

    // create a collection
    const std::string field_id = "id";
    const std::string field_vector = "vector";
    const std::string field_json = "json_field";
    const uint32_t dimension = 128;

    // collection schema, create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "user id", true, true});
    collection_schema.AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(dimension));
    collection_schema.AddField({field_json, milvus::DataType::JSON, "properties"});

    status = client->CreateCollection(collection_schema);
    util::CheckStatus("Failed to create collection:", status);
    std::cout << "Successfully create collection " << collection_name << std::endl;

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::COSINE);
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("Failed to create index on vector field:", status);
    std::cout << "Successfully create index." << std::endl;

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("Failed to load collection:", status);

    // insert some rows
    const int64_t row_count = 10;
    std::vector<nlohmann::json> rows;
    for (auto i = 0; i < row_count; ++i) {
        nlohmann::json row;
        row[field_json] = {{"age", util::RandomeValue<int>(1, 100)}, {"name", "user_" + std::to_string(i)}};
        row[field_vector] = util::GenerateFloatVector(dimension);
        rows.emplace_back(std::move(row));
    }

    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", rows, dml_results);
    util::CheckStatus("Failed to insert:", status);
    std::cout << "Successfully insert " << dml_results.InsertCount() << " rows." << std::endl;

    // query
    milvus::QueryArguments q_arguments{};
    q_arguments.SetCollectionName(collection_name);
    q_arguments.AddOutputField(field_id);
    q_arguments.AddOutputField(field_json);
    q_arguments.SetLimit(5);
    // set to strong level so that the query is executed after the inserted data is consumed by server
    q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResults query_resutls{};
    status = client->Query(q_arguments, query_resutls);
    util::CheckStatus("Failed to query:", status);
    std::cout << "Successfully query." << std::endl;

    std::vector<nlohmann::json> output_rows;
    status = query_resutls.OutputRows(output_rows);
    util::CheckStatus("Failed to get output rows:", status);
    std::cout << "Query results:" << std::endl;
    for (const auto& row : output_rows) {
        std::cout << "\t" << row << std::endl;
    }

    // do search
    milvus::SearchArguments s_arguments{};
    s_arguments.SetCollectionName(collection_name);
    s_arguments.SetLimit(3);
    s_arguments.AddOutputField(field_id);
    s_arguments.AddOutputField(field_json);

    auto q_number_1 = util::RandomeValue<int64_t>(0, row_count - 1);
    auto q_number_2 = util::RandomeValue<int64_t>(0, row_count - 1);
    s_arguments.AddFloatVector(field_vector, rows[q_number_1][field_vector]);
    s_arguments.AddFloatVector(field_vector, rows[q_number_2][field_vector]);
    std::cout << "Searching the No." << q_number_1 << " and No." << q_number_2 << std::endl;

    milvus::SearchResults search_results{};
    status = client->Search(s_arguments, search_results);
    util::CheckStatus("Failed to search:", status);
    std::cout << "Successfully search." << std::endl;

    for (auto& result : search_results.Results()) {
        std::cout << "Result of one target vector:" << std::endl;
        std::vector<nlohmann::json> output_rows;
        status = result.OutputRows(output_rows);
        util::CheckStatus("Failed to get output rows:", status);
        for (const auto& row : output_rows) {
            std::cout << "\t" << row << std::endl;
        }
    }

    client->Disconnect();
    return 0;
}
