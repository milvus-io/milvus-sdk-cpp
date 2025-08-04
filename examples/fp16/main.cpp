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
    const std::string collection_name = "TEST_CPP_FP16";
    status = client->DropCollection(collection_name);

    // create a collection
    const std::string field_id = "pk";
    const std::string field_vector_fp16 = "vector_fp16";
    const std::string field_vector_bf16 = "vector_bf16";
    const std::string field_text = "text";
    const uint32_t dimension = 4;

    // collection schema, create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "id", true, false});
    collection_schema.AddField(
        milvus::FieldSchema(field_vector_fp16, milvus::DataType::FLOAT16_VECTOR).WithDimension(dimension));
    collection_schema.AddField(
        milvus::FieldSchema(field_vector_bf16, milvus::DataType::BFLOAT16_VECTOR).WithDimension(dimension));
    collection_schema.AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(100));

    status = client->CreateCollection(collection_schema);
    util::CheckStatus("Failed to create collection:", status);
    std::cout << "Successfully create collection " << collection_name << std::endl;

    // create index
    milvus::IndexDesc index_vector_fp16(field_vector_fp16, "", milvus::IndexType::AUTOINDEX,
                                        milvus::MetricType::COSINE);
    status = client->CreateIndex(collection_name, index_vector_fp16);
    util::CheckStatus("Failed to create index on float16 vector field:", status);
    milvus::IndexDesc index_vector_bf16(field_vector_bf16, "", milvus::IndexType::AUTOINDEX,
                                        milvus::MetricType::COSINE);
    status = client->CreateIndex(collection_name, index_vector_bf16);
    util::CheckStatus("Failed to create index on bfloat16 vector field:", status);
    std::cout << "Successfully create index." << std::endl;

    // insert some rows
    const int64_t row_count = 100;
    std::vector<int64_t> insert_ids(row_count);
    std::vector<std::vector<float>> src_vectors_fp16 = util::GenerateFloatVectors(dimension, row_count);
    std::vector<std::vector<float>> src_vectors_bf16 = util::GenerateFloatVectors(dimension, row_count);
    std::vector<std::vector<uint16_t>> insert_vectors_fp16;
    std::vector<std::vector<uint16_t>> insert_vectors_bf16;
    std::vector<std::string> insert_texts(row_count);
    for (auto i = 0; i < row_count; ++i) {
        insert_ids[i] = i;
        insert_texts[i] = "hello world " + std::to_string(i);
        insert_vectors_fp16.emplace_back(util::GenerateFloat16Vector(src_vectors_fp16[i]));
        insert_vectors_bf16.emplace_back(util::GenerateBFloat16Vector(src_vectors_bf16[i]));
    }

    std::vector<milvus::FieldDataPtr> fields_data{
        std::make_shared<milvus::Int64FieldData>(field_id, insert_ids),
        std::make_shared<milvus::VarCharFieldData>(field_text, insert_texts),
        std::make_shared<milvus::Float16VecFieldData>(field_vector_fp16, insert_vectors_fp16),
        std::make_shared<milvus::BFloat16VecFieldData>(field_vector_bf16, insert_vectors_bf16),
    };
    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", fields_data, dml_results);
    util::CheckStatus("Failed to insert:", status);
    std::cout << "Successfully insert " << dml_results.IdArray().IntIDArray().size() << " rows." << std::endl;
    const auto& ids = dml_results.IdArray().IntIDArray();

    // load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("Failed to load collection:", status);
    std::cout << "Successfully load collection." << std::endl;

    // print the original vector data
    int pk_1 = 10, pk_2 = 50;
    std::cout << "Original " << field_vector_fp16 << " at " << pk_1 << ": ";
    util::PrintList(src_vectors_fp16[pk_1]);
    std::cout << std::endl;
    std::cout << "Original " << field_vector_bf16 << " at " << pk_1 << ": ";
    util::PrintList(src_vectors_bf16[pk_1]);
    std::cout << std::endl;

    std::cout << "Original " << field_vector_fp16 << " at " << pk_2 << ": ";
    util::PrintList(src_vectors_fp16[pk_2]);
    std::cout << std::endl;
    std::cout << "Original " << field_vector_bf16 << " at " << pk_2 << ": ";
    util::PrintList(src_vectors_bf16[pk_2]);
    std::cout << std::endl;

    {
        // query
        std::string expr = field_id + " in [" + std::to_string(pk_1) + "," + std::to_string(pk_2) + "]";
        milvus::QueryArguments q_arguments{};
        q_arguments.SetCollectionName(collection_name);
        q_arguments.SetFilter(expr);
        q_arguments.AddOutputField(field_id);
        q_arguments.AddOutputField(field_text);
        q_arguments.AddOutputField(field_vector_fp16);
        q_arguments.AddOutputField(field_vector_bf16);
        q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        std::cout << "Query with expression: " << expr << std::endl;
        milvus::QueryResults query_resutls{};
        status = client->Query(q_arguments, query_resutls);
        util::CheckStatus("Failed to query:", status);
        std::cout << "Successfully query." << std::endl;

        auto id_field_data = query_resutls.OutputField<milvus::Int64FieldData>(field_id);
        auto text_field_data = query_resutls.OutputField<milvus::VarCharFieldData>(field_text);
        auto vetor_fp16_field_data = query_resutls.OutputField<milvus::Float16VecFieldData>(field_vector_fp16);
        auto vetor_bf16_field_data = query_resutls.OutputField<milvus::BFloat16VecFieldData>(field_vector_bf16);

        for (size_t i = 0; i < id_field_data->Count(); ++i) {
            std::cout << "\t" << field_id << ":" << id_field_data->Value(i) << "\t" << field_text << ":"
                      << text_field_data->Value(i);

            std::cout << "\t" << field_vector_fp16 << ":";
            util::PrintListF16AsF32(vetor_fp16_field_data->Value(i), true);

            std::cout << "\t" << field_vector_bf16 << ":";
            util::PrintListF16AsF32(vetor_bf16_field_data->Value(i), false);
            std::cout << std::endl;
        }
    }

    {
        // search
        milvus::SearchArguments s_arguments{};
        s_arguments.SetCollectionName(collection_name);
        s_arguments.SetLimit(3);
        s_arguments.AddOutputField(field_vector_fp16);
        // set to BOUNDED level to accept data inconsistence within a time window(default is 5 seconds)
        s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        s_arguments.AddFloat16Vector(field_vector_fp16, insert_vectors_fp16[pk_1]);
        s_arguments.AddFloat16Vector(field_vector_fp16, insert_vectors_fp16[pk_2]);
        std::cout << "Searching the No." << pk_1 << " and Nop," << pk_2 << " vectors." << std::endl;

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

            auto vetor_fp16_field_data = result.OutputField<milvus::Float16VecFieldData>(field_vector_fp16);
            for (size_t i = 0; i < ids.size(); ++i) {
                std::cout << "\t" << result.PrimaryKeyName() << ":" << ids[i] << "\tDistance: " << distances[i];
                std::cout << "\t" << field_vector_fp16 << ":";
                util::PrintListF16AsF32(vetor_fp16_field_data->Value(i), true);
                std::cout << std::endl;
            }
        }
    }

    client->Disconnect();
    return 0;
}
