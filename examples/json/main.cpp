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
    CheckStatus("Failed to connect milvus server:", status);
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
    CheckStatus("Failed to create collection:", status);
    std::cout << "Successfully create collection." << std::endl;

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::COSINE);
    status = client->CreateIndex(collection_name, index_vector);
    CheckStatus("Failed to create index on vector field:", status);
    std::cout << "Successfully create index." << std::endl;

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    CheckStatus("Failed to load collection:", status);

    // insert some rows
    const int64_t row_count = 10;
    std::vector<nlohmann::json> insert_jsons(row_count);
    std::vector<std::vector<float>> insert_vectors = GenerateFloatVectors(dimension, row_count);
    for (auto i = 0; i < row_count; ++i) {
        nlohmann::json obj = {{"age", RandomeValue<int>(1, 100)}, {"name", "user_" + std::to_string(i)}};
        insert_jsons[i] = obj;
    }

    std::vector<milvus::FieldDataPtr> fields_data{
        std::make_shared<milvus::JSONFieldData>(field_json, insert_jsons),
        std::make_shared<milvus::FloatVecFieldData>(field_vector, insert_vectors)};
    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", fields_data, dml_results);
    CheckStatus("Failed to insert:", status);
    std::cout << "Successfully insert " << dml_results.IdArray().IntIDArray().size() << " rows." << std::endl;

    // query
    milvus::QueryArguments q_arguments{};
    q_arguments.SetCollectionName(collection_name);
    q_arguments.AddOutputField(field_id);
    q_arguments.AddOutputField(field_json);
    q_arguments.SetLimit(5);

    milvus::QueryResults query_resutls{};
    status = client->Query(q_arguments, query_resutls);
    CheckStatus("Failed to query:", status);
    std::cout << "Successfully query." << std::endl;

    for (auto& field_data : query_resutls.OutputFields()) {
        std::cout << "Field: " << field_data->Name() << " Count:" << field_data->Count() << std::endl;
    }

    // do search
    milvus::SearchArguments s_arguments{};
    s_arguments.SetCollectionName(collection_name);
    s_arguments.SetTopK(3);
    s_arguments.AddOutputField(field_id);
    s_arguments.AddOutputField(field_json);

    auto q_number_1 = RandomeValue<int64_t>(0, row_count - 1);
    auto q_number_2 = RandomeValue<int64_t>(0, row_count - 1);
    s_arguments.AddTargetVector(field_vector, std::move(insert_vectors[q_number_1]));
    s_arguments.AddTargetVector(field_vector, std::move(insert_vectors[q_number_2]));
    std::cout << "Searching the No." << q_number_1 << " and No." << q_number_2 << std::endl;

    milvus::SearchResults search_results{};
    status = client->Search(s_arguments, search_results);
    CheckStatus("Failed to search:", status);
    std::cout << "Successfully search." << std::endl;

    for (auto& result : search_results.Results()) {
        auto& ids = result.Ids().IntIDArray();
        auto& distances = result.Scores();
        if (ids.size() != distances.size()) {
            std::cout << "Illegal result!" << std::endl;
            continue;
        }

        std::cout << "Result of one target vector:" << std::endl;

        auto id_field = result.OutputField(field_id);
        milvus::Int64FieldDataPtr id_field_ptr = std::static_pointer_cast<milvus::Int64FieldData>(id_field);
        auto& id_data = id_field_ptr->Data();

        auto json_field = result.OutputField(field_json);
        milvus::JSONFieldDataPtr json_field_ptr = std::static_pointer_cast<milvus::JSONFieldData>(json_field);
        auto& json_data = json_field_ptr->Data();

        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << "\tID: " << ids[i] << "\tDistance: " << distances[i] << "\tID: " << id_data[i]
                      << "\tJSON: " << json_data[i].dump() << std::endl;
        }
    }

    client->Disconnect();
    return 0;
}
