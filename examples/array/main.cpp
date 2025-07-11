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

#include "milvus/MilvusClient.h"
#include "milvus/types/CollectionSchema.h"
#include "util/ExampleUtils.h"

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("Failed to connect milvus server:", status);
    std::cout << "Connect to milvus server." << std::endl;

    // drop the collection if it exists
    const std::string collection_name = "TEST_CPP_ARRAY";
    status = client->DropCollection(collection_name);

    // create a collection
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

    // collection schema, create collection
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
    milvus::VarCharFieldDataPtr id_field = std::make_shared<milvus::VarCharFieldData>(field_id);
    milvus::FloatVecFieldDataPtr vector_field = std::make_shared<milvus::FloatVecFieldData>(field_vector);
    milvus::ArrayBoolFieldDataPtr arr_bool_field = std::make_shared<milvus::ArrayBoolFieldData>(field_array_bool);
    milvus::ArrayInt8FieldDataPtr arr_int8_field = std::make_shared<milvus::ArrayInt8FieldData>(field_array_int8);
    milvus::ArrayInt16FieldDataPtr arr_int16_field = std::make_shared<milvus::ArrayInt16FieldData>(field_array_int16);
    milvus::ArrayInt32FieldDataPtr arr_int32_field = std::make_shared<milvus::ArrayInt32FieldData>(field_array_int32);
    milvus::ArrayInt64FieldDataPtr arr_int64_field = std::make_shared<milvus::ArrayInt64FieldData>(field_array_int64);
    milvus::ArrayFloatFieldDataPtr arr_float_field = std::make_shared<milvus::ArrayFloatFieldData>(field_array_float);
    milvus::ArrayDoubleFieldDataPtr arr_double_field =
        std::make_shared<milvus::ArrayDoubleFieldData>(field_array_double);
    milvus::ArrayVarCharFieldDataPtr arr_varchar_field =
        std::make_shared<milvus::ArrayVarCharFieldData>(field_array_varchar);

    const int64_t row_count = 10;
    for (auto i = 0; i < row_count; ++i) {
        id_field->Add("user_" + std::to_string(i));
        vector_field->Add(util::GenerateFloatVector(dimension));

        auto cap = util::RandomeValue<int>(1, 5);
        arr_bool_field->Add(util::RansomBools(cap));
        arr_int8_field->Add(util::RandomeValues<int8_t>(0, 100, cap));
        arr_int16_field->Add(util::RandomeValues<int16_t>(0, 1000, cap));
        arr_int32_field->Add(util::RandomeValues<int32_t>(0, 10000, cap));
        arr_int64_field->Add(util::RandomeValues<int64_t>(0, 100000, cap));
        arr_float_field->Add(util::RandomeValues<float>(0.0, 1.0, cap));
        arr_double_field->Add(util::RandomeValues<double>(0.0, 10.0, cap));

        auto values = util::RandomeValues<int>(0, 100, cap);
        std::vector<std::string> varchars(cap);
        std::transform(values.begin(), values.end(), varchars.begin(),
                       [i](int x) { return "varchar_" + std::to_string(i * 10000 + x); });
        arr_varchar_field->Add(varchars);
    }

    std::vector<milvus::FieldDataPtr> fields_data{id_field,         vector_field,     arr_bool_field,  arr_int8_field,
                                                  arr_int16_field,  arr_int32_field,  arr_int64_field, arr_float_field,
                                                  arr_double_field, arr_varchar_field};
    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", fields_data, dml_results);
    util::CheckStatus("Failed to insert:", status);
    std::cout << "Successfully insert " << dml_results.IdArray().StrIDArray().size() << " rows." << std::endl;

    // query
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

    milvus::QueryResults query_resutls{};
    status = client->Query(q_arguments, query_resutls);
    util::CheckStatus("Failed to query:", status);
    std::cout << "Successfully query." << std::endl;

    for (auto& field_data : query_resutls.OutputFields()) {
        std::cout << "Field: " << field_data->Name() << " Count:" << field_data->Count() << std::endl;
    }

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
    s_arguments.AddFloatVector(field_vector, vector_field->Data()[q_number_1]);
    s_arguments.AddFloatVector(field_vector, vector_field->Data()[q_number_2]);
    std::cout << "Searching the No." << q_number_1 << " and No." << q_number_2 << std::endl;

    milvus::SearchResults search_results{};
    status = client->Search(s_arguments, search_results);
    util::CheckStatus("Failed to search:", status);
    std::cout << "Successfully search." << std::endl;

    for (auto& result : search_results.Results()) {
        auto& ids = result.Ids().StrIDArray();
        auto& distances = result.Scores();
        if (ids.size() != distances.size()) {
            std::cout << "Illegal result!" << std::endl;
            continue;
        }

        std::cout << "Result of one target vector:" << std::endl;

        auto id_field = result.OutputField<milvus::VarCharFieldData>(field_id);
        auto array_int16_field = result.OutputField<milvus::ArrayInt16FieldData>(field_array_int16);
        auto array_varchar_field = result.OutputField<milvus::ArrayVarCharFieldData>(field_array_varchar);
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << "\t" << id_field->Name() << ":" << ids[i] << "\tDistance: " << distances[i] << "\t";
            std::cout << array_int16_field->Name();
            util::PrintList<int16_t>(array_int16_field->Value(i));

            std::cout << array_varchar_field->Name();
            util::PrintList<std::string>(array_varchar_field->Value(i));
            std::cout << std::endl;
        }
    }

    client->Disconnect();
    return 0;
}
