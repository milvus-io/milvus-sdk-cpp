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

    const std::string collection_name = "CPP_V1_ARRAY";
    const std::string field_id = "id";
    const std::string field_vector = "vector";
    const std::string field_array_bool = "field_array_bool";
    const std::string field_array_int8 = "field_array_int8";
    const std::string field_array_int16 = "array_int16_field";
    const std::string field_array_int32 = "field_array_int32";
    const std::string field_array_int64 = "field_array_int64";
    const std::string field_array_float = "field_array_float";
    const std::string field_array_double = "field_array_double";
    const std::string field_array_varchar = "field_array_varchar";

    const uint32_t dimension = 128;

    // collection schema, drop and create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField(
        milvus::FieldSchema(field_id, milvus::DataType::VARCHAR, "user id", true, false).WithMaxLength(64));
    collection_schema.AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(dimension));
    collection_schema.AddField(milvus::FieldSchema(field_array_bool, milvus::DataType::ARRAY, "bool array")
                                   .WithMaxCapacity(10)
                                   .WithElementType(milvus::DataType::BOOL));
    collection_schema.AddField(milvus::FieldSchema(field_array_int8, milvus::DataType::ARRAY, "int8 array")
                                   .WithMaxCapacity(10)
                                   .WithElementType(milvus::DataType::INT8));
    collection_schema.AddField(milvus::FieldSchema(field_array_int16, milvus::DataType::ARRAY, "int16 array")
                                   .WithMaxCapacity(10)
                                   .WithElementType(milvus::DataType::INT16));
    collection_schema.AddField(milvus::FieldSchema(field_array_int32, milvus::DataType::ARRAY, "int32 array")
                                   .WithMaxCapacity(10)
                                   .WithElementType(milvus::DataType::INT32));
    collection_schema.AddField(milvus::FieldSchema(field_array_int64, milvus::DataType::ARRAY, "int64 array")
                                   .WithMaxCapacity(10)
                                   .WithElementType(milvus::DataType::INT64));
    collection_schema.AddField(milvus::FieldSchema(field_array_float, milvus::DataType::ARRAY, "float array")
                                   .WithMaxCapacity(10)
                                   .WithElementType(milvus::DataType::FLOAT));
    collection_schema.AddField(milvus::FieldSchema(field_array_double, milvus::DataType::ARRAY, "double array")
                                   .WithMaxCapacity(10)
                                   .WithElementType(milvus::DataType::DOUBLE));
    collection_schema.AddField(milvus::FieldSchema(field_array_varchar, milvus::DataType::ARRAY, "string array")
                                   .WithElementType(milvus::DataType::VARCHAR)
                                   .WithMaxCapacity(100)
                                   .WithMaxLength(1024));

    status = client->DropCollection(collection_name);
    status = client->CreateCollection(collection_schema);
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::COSINE);
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("load collection: " + collection_name, status);

    // insert some rows
    const int64_t row_count = 10;
    milvus::EntityRows rows;
    for (auto i = 0; i < row_count; ++i) {
        auto cap = util::RandomeValue<int>(1, 5);
        milvus::EntityRow row;
        row[field_id] = "user_" + std::to_string(i);
        row[field_vector] = util::GenerateFloatVector(dimension);
        row[field_array_bool] = util::RansomBools(cap);
        row[field_array_int8] = util::RandomeValues<int8_t>(0, 100, cap);
        row[field_array_int16] = util::RandomeValues<int16_t>(0, 1000, cap);
        row[field_array_int32] = util::RandomeValues<int32_t>(0, 10000, cap);
        row[field_array_int64] = util::RandomeValues<int64_t>(0, 100000, cap);
        row[field_array_float] = util::RandomeValues<float>(0.0, 1.0, cap);
        row[field_array_double] = util::RandomeValues<double>(0.0, 10.0, cap);
        std::vector<std::string> varchars(cap);
        auto values = util::RandomeValues<int>(0, 100, cap);
        std::transform(values.begin(), values.end(), varchars.begin(),
                       [i](int x) { return "varchar_" + std::to_string(i * 10000 + x); });
        row[field_array_varchar] = varchars;
        rows.emplace_back(std::move(row));
    }

    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", rows, dml_results);
    util::CheckStatus("insert", status);
    std::cout << dml_results.IdArray().StrIDArray().size() << " rows inserted." << std::endl;

    {
        // query some items wihtout filtering
        milvus::QueryArguments q_arguments{};
        q_arguments.SetCollectionName(collection_name);
        q_arguments.AddOutputField(field_id);
        q_arguments.AddOutputField(field_array_bool);
        q_arguments.AddOutputField(field_array_int8);
        q_arguments.AddOutputField(field_array_int16);
        q_arguments.AddOutputField(field_array_int32);
        q_arguments.AddOutputField(field_array_int64);
        q_arguments.AddOutputField(field_array_float);
        q_arguments.AddOutputField(field_array_double);
        q_arguments.AddOutputField(field_array_varchar);
        q_arguments.SetLimit(5);
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
    }

    {
        // do search
        milvus::SearchArguments s_arguments{};
        s_arguments.SetCollectionName(collection_name);
        s_arguments.SetLimit(3);
        s_arguments.AddOutputField(field_id);
        s_arguments.AddOutputField(field_array_bool);
        s_arguments.AddOutputField(field_array_int8);
        s_arguments.AddOutputField(field_array_int16);
        s_arguments.AddOutputField(field_array_int32);
        s_arguments.AddOutputField(field_array_int64);
        s_arguments.AddOutputField(field_array_float);
        s_arguments.AddOutputField(field_array_double);
        s_arguments.AddOutputField(field_array_varchar);

        auto q_number_1 = util::RandomeValue<int64_t>(0, row_count - 1);
        auto q_number_2 = util::RandomeValue<int64_t>(0, row_count - 1);
        s_arguments.AddFloatVector(field_vector, rows[q_number_1][field_vector]);
        s_arguments.AddFloatVector(field_vector, rows[q_number_2][field_vector]);
        std::cout << "Searching the No." << q_number_1 << " and No." << q_number_2 << std::endl;

        milvus::SearchResults search_results{};
        status = client->Search(s_arguments, search_results);
        util::CheckStatus("search", status);

        for (auto& result : search_results.Results()) {
            std::cout << "Result of one target vector:" << std::endl;
            milvus::EntityRows output_rows;
            status = result.OutputRows(output_rows);
            util::CheckStatus("get output rows", status);
            std::cout << "Result of one target vector:" << std::endl;
            for (const auto& row : output_rows) {
                std::cout << "\t" << row << std::endl;
            }
        }
    }

    client->Disconnect();
    return 0;
}
