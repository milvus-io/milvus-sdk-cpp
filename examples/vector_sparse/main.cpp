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
    const std::string collection_name = "TEST_CPP_SPARSE";
    status = client->DropCollection(collection_name);

    // create a collection
    const std::string field_id = "pk";
    const std::string field_vector = "sparse";
    const std::string field_text = "text";

    // collection schema, create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "user id", true, false});
    collection_schema.AddField({field_vector, milvus::DataType::SPARSE_FLOAT_VECTOR});
    collection_schema.AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(1024));

    status = client->CreateCollection(collection_schema);
    util::CheckStatus("Failed to create collection:", status);
    std::cout << "Successfully create collection " << collection_name << std::endl;

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::SPARSE_INVERTED_INDEX, milvus::MetricType::IP);
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("Failed to create index on vector field:", status);
    std::cout << "Successfully create index." << std::endl;

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("Failed to load collection:", status);

    // insert some rows
    const int64_t row_count = 10;
    std::vector<int64_t> ids(row_count);
    auto insert_vectors = util::GenerateSparseVectors(10, row_count);
    std::vector<std::string> texts(row_count);
    for (auto i = 0; i < row_count; ++i) {
        ids[i] = i;
        texts[i] = "this is text_" + std::to_string(i);
    }

    std::vector<milvus::FieldDataPtr> fields_data{
        std::make_shared<milvus::Int64FieldData>(field_id, ids),
        std::make_shared<milvus::SparseFloatVecFieldData>(field_vector, insert_vectors),
        std::make_shared<milvus::VarCharFieldData>(field_text, texts)};
    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", fields_data, dml_results);
    util::CheckStatus("Failed to insert:", status);
    std::cout << "Successfully insert " << dml_results.IdArray().IntIDArray().size() << " rows." << std::endl;

    // query
    milvus::QueryArguments q_arguments{};
    q_arguments.SetCollectionName(collection_name);
    q_arguments.AddOutputField(field_vector);
    q_arguments.AddOutputField(field_text);
    q_arguments.SetLimit(5);
    // set to strong level so that the query is executed after the inserted data is consumed by server
    q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResults query_resutls{};
    status = client->Query(q_arguments, query_resutls);
    util::CheckStatus("Failed to query:", status);
    std::cout << "Successfully query." << std::endl;

    auto id_field_data = query_resutls.OutputField<milvus::Int64FieldData>(field_id);
    auto text_field_data = query_resutls.OutputField<milvus::VarCharFieldData>(field_text);
    auto vetor_field_data = query_resutls.OutputField<milvus::SparseFloatVecFieldData>(field_vector);

    for (size_t i = 0; i < id_field_data->Count(); ++i) {
        std::cout << "\t" << id_field_data->Name() << ":" << id_field_data->Value(i) << "\t" << text_field_data->Name()
                  << ":" << text_field_data->Value(i) << "\t" << vetor_field_data->Name() << ":";
        util::PrintMap(vetor_field_data->Value(i));
        std::cout << std::endl;
    }

    // do search
    milvus::SearchArguments s_arguments{};
    s_arguments.SetCollectionName(collection_name);
    s_arguments.SetLimit(3);
    s_arguments.AddOutputField(field_vector);
    s_arguments.AddOutputField(field_text);

    auto q_number_1 = util::RandomeValue<int64_t>(0, row_count - 1);
    auto q_number_2 = util::RandomeValue<int64_t>(0, row_count - 1);
    s_arguments.AddSparseVector(field_vector, insert_vectors[q_number_1]);
    s_arguments.AddSparseVector(field_vector, insert_vectors[q_number_2]);
    s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    std::cout << "Searching the ID." << q_number_1 << " sparse vector: ";
    util::PrintMap(insert_vectors[q_number_1]);
    std::cout << std::endl;
    std::cout << "Searching the ID." << q_number_2 << " sparse vector: ";
    util::PrintMap(insert_vectors[q_number_2]);
    std::cout << std::endl;

    milvus::SearchResults search_results{};
    status = client->Search(s_arguments, search_results);
    util::CheckStatus("Failed to search:", status);
    std::cout << "Successfully search." << std::endl;

    for (auto& result : search_results.Results()) {
        auto& ids = result.Ids().IntIDArray();
        auto& distances = result.Scores();
        if (ids.size() != distances.size()) {
            std::cout << "Illegal result!" << std::endl;
            continue;
        }

        std::cout << "Result of one target vector:" << std::endl;

        auto text_field = result.OutputField<milvus::VarCharFieldData>(field_text);
        auto vector_field = result.OutputField<milvus::SparseFloatVecFieldData>(field_vector);
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << "\t" << result.PrimaryKeyName() << ":" << ids[i] << "\tDistance: " << distances[i] << "\t"
                      << text_field->Name() << ":" << text_field->Value(i) << "\t" << vector_field->Name() << ":";
            util::PrintMap(vector_field->Value(i));
            std::cout << std::endl;
        }
    }

    client->Disconnect();
    return 0;
}
