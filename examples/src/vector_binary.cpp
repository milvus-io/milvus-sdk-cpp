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

    const std::string collection_name = "TEST_CPP_BINARY";
    const std::string field_id = "pk";
    const std::string field_vector = "vector";
    const std::string field_text = "text";
    const uint32_t dimension = 128;

    // collection schema, drop and create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField(
        milvus::FieldSchema(field_id, milvus::DataType::VARCHAR, "", true, false).WithMaxLength(128));
    collection_schema.AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::BINARY_VECTOR).WithDimension(dimension));
    collection_schema.AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(1024));

    status = client->DropCollection(collection_name);
    status = client->CreateCollection(collection_schema);
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::BIN_IVF_FLAT, milvus::MetricType::HAMMING);
    index_vector.AddExtraParam(milvus::NLIST, "5");
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("load collection: " + collection_name, status);

    {
        // insert some rows by column-based
        auto ids = std::vector<std::string>{"primary_key_10000", "primary_key_10001"};
        auto texts = std::vector<std::string>{"column-based-1", "column-based-2"};
        auto vectors = util::GenerateBinaryVectors(dimension, 2);
        std::vector<milvus::FieldDataPtr> fields_data{
            std::make_shared<milvus::VarCharFieldData>(field_id, ids),
            std::make_shared<milvus::VarCharFieldData>(field_text, texts),
            std::make_shared<milvus::BinaryVecFieldData>(field_vector, vectors)};
        milvus::DmlResults dml_results;
        status = client->Insert(collection_name, "", fields_data, dml_results);
        util::CheckStatus("insert", status);
        std::cout << dml_results.InsertCount() << " rows inserted by column-based." << std::endl;
    }

    const int64_t row_count = 10;
    milvus::EntityRows rows;
    {
        // insert some rows
        for (auto i = 0; i < row_count; ++i) {
            milvus::EntityRow row;
            row[field_id] = "primary_key_" + std::to_string(i);
            row[field_text] = "this is text_" + std::to_string(i);
            row[field_vector] = util::GenerateBinaryVector(dimension);
            rows.emplace_back(std::move(row));
        }

        milvus::DmlResults dml_results;
        status = client->Insert(collection_name, "", rows, dml_results);
        util::CheckStatus("insert", status);
        std::cout << dml_results.InsertCount() << " rows inserted by row-based." << std::endl;
    }

    // query
    auto q_number_1 = util::RandomeValue<int64_t>(0, row_count - 1);
    auto q_number_2 = util::RandomeValue<int64_t>(0, row_count - 1);
    auto q_id_1 = rows[q_number_1][field_id].get<std::string>();
    auto q_id_2 = rows[q_number_2][field_id].get<std::string>();
    std::string filter = field_id + " in [\"" + q_id_1 + "\", \"" + q_id_2 + "\"]";
    std::cout << "Query with filter expression: " << filter << std::endl;

    milvus::QueryArguments q_arguments{};
    q_arguments.SetCollectionName(collection_name);
    q_arguments.AddOutputField(field_vector);
    q_arguments.AddOutputField(field_text);
    q_arguments.SetFilter(filter);
    // set to strong level so that the query is executed after the inserted data is consumed by server
    q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

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

    // do search
    auto q_vector_1 = rows[q_number_1][field_vector];
    auto q_vector_2 = rows[q_number_2][field_vector];
    milvus::SearchArguments s_arguments{};
    s_arguments.SetCollectionName(collection_name);
    s_arguments.SetLimit(3);
    s_arguments.AddOutputField(field_vector);
    s_arguments.AddOutputField(field_text);
    s_arguments.AddBinaryVector(field_vector, q_vector_1.get<std::vector<uint8_t>>());
    s_arguments.AddBinaryVector(field_vector, q_vector_2.get<std::vector<uint8_t>>());
    s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    std::cout << "Searching the ID." << q_number_1 << " binary vector: " << q_vector_1 << std::endl;
    std::cout << "Searching the ID." << q_number_2 << " binary vector: " << q_vector_2 << std::endl;

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

    client->Disconnect();
    return 0;
}
