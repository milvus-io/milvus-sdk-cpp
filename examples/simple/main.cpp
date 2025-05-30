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
#include <random>
#include <string>
#include <thread>

#include "milvus/MilvusClient.h"
#include "milvus/types/CollectionSchema.h"

void
CheckStatus(std::string&& prefix, const milvus::Status& status) {
    if (!status.IsOk()) {
        std::cout << prefix << " " << status.Message() << std::endl;
        exit(1);
    }
}

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530};
    auto status = client->Connect(connect_param);
    CheckStatus("Failed to connect milvus server:", status);
    std::cout << "Connect to milvus server." << std::endl;

    // drop the collection if it exists
    const std::string collection_name = "TEST";
    status = client->DropCollection(collection_name);

    // create a collection
    const std::string field_id = "user_id";
    const std::string field_name = "user_name";
    const std::string field_age = "user_age";
    const std::string field_face = "user_face";
    const uint32_t dimension = 128;
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "user id", true, false});
    milvus::FieldSchema varchar_scheam{field_name, milvus::DataType::VARCHAR, "user name"};
    varchar_scheam.SetMaxLength(100);
    collection_schema.AddField(varchar_scheam);
    collection_schema.AddField({field_age, milvus::DataType::INT8, "user age"});
    collection_schema.AddField(
        milvus::FieldSchema(field_face, milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(dimension));

    status = client->CreateCollection(collection_schema);
    CheckStatus("Failed to create collection:", status);
    std::cout << "Successfully create collection." << std::endl;

    // create index (required after 2.2.0)
    milvus::IndexDesc index_vector(field_face, "", milvus::IndexType::FLAT, milvus::MetricType::COSINE);
    status = client->CreateIndex(collection_name, index_vector);
    CheckStatus("Failed to create index on vector field:", status);
    std::cout << "Successfully create index." << std::endl;

    milvus::IndexDesc index_varchar(field_name, "", milvus::IndexType::TRIE);
    status = client->CreateIndex(collection_name, index_varchar);
    CheckStatus("Failed to create index on varchar field:", status);
    std::cout << "Successfully create index." << std::endl;

    milvus::IndexDesc index_sort(field_age, "", milvus::IndexType::STL_SORT);
    status = client->CreateIndex(collection_name, index_sort);
    CheckStatus("Failed to create index on integer field:", status);
    std::cout << "Successfully create index." << std::endl;

    // create a partition
    std::string partition_name = "Year_2022";
    status = client->CreatePartition(collection_name, partition_name);
    CheckStatus("Failed to create partition:", status);
    std::cout << "Successfully create partition." << std::endl;

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    CheckStatus("Failed to load collection:", status);

    // insert some rows
    const int64_t row_count = 1000;
    std::vector<int64_t> insert_ids;
    std::vector<std::string> insert_names;
    std::vector<int8_t> insert_ages;
    std::vector<std::vector<float>> insert_vectors;
    std::default_random_engine ran(time(nullptr));
    std::uniform_int_distribution<int> int_gen(1, 100);
    std::uniform_real_distribution<float> float_gen(0.0, 1.0);
    for (auto i = 0; i < row_count; ++i) {
        insert_ids.push_back(i);
        insert_names.push_back("user_" + std::to_string(i));
        insert_ages.push_back(int_gen(ran));
        std::vector<float> vector(dimension);

        for (auto i = 0; i < dimension; ++i) {
            vector[i] = float_gen(ran);
        }
        insert_vectors.emplace_back(vector);
    }

    std::vector<milvus::FieldDataPtr> fields_data{
        std::make_shared<milvus::Int64FieldData>(field_id, insert_ids),
        std::make_shared<milvus::VarCharFieldData>(field_name, insert_names),
        std::make_shared<milvus::Int8FieldData>(field_age, insert_ages),
        std::make_shared<milvus::FloatVecFieldData>(field_face, insert_vectors)};
    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, partition_name, fields_data, dml_results);
    CheckStatus("Failed to insert:", status);
    std::cout << "Successfully insert " << dml_results.IdArray().IntIDArray().size() << " rows." << std::endl;

    // flush
    status = client->Flush(std::vector<std::string>{collection_name});
    CheckStatus("Failed to flush:", status);
    std::cout << "Flushed" << std::endl;

    // get partition statistics
    milvus::PartitionStat part_stat;
    status = client->GetPartitionStatistics(collection_name, partition_name, part_stat);
    CheckStatus("Failed to get partition statistics:", status);
    std::cout << "Partition " << partition_name << " row count: " << part_stat.RowCount() << std::endl;

    // delete
    milvus::DmlResults del_res;
    status = client->Delete(collection_name, partition_name, field_id + "== 5", del_res);
    CheckStatus("Failed to delete entity:", status);
    std::cout << "Delete entity whose id is 5" << std::endl;

    // query
    milvus::QueryArguments q_arguments{};
    q_arguments.SetCollectionName(collection_name);
    q_arguments.AddPartitionName(partition_name);
    q_arguments.SetExpression(field_id + " in [5, 10]");
    q_arguments.AddOutputField(field_id);
    q_arguments.AddOutputField(field_name);
    q_arguments.AddOutputField(field_age);

    std::cout << "Query with expression: " << q_arguments.Expression() << std::endl;
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
    s_arguments.AddPartitionName(partition_name);
    s_arguments.SetTopK(10);
    s_arguments.AddOutputField(field_name);
    s_arguments.AddOutputField(field_age);
    std::string filter_expr = field_age + " > 40";
    s_arguments.SetExpression(filter_expr);
    // set to strong guarantee so that the search is executed after the inserted data is persisted
    s_arguments.SetGuaranteeTimestamp(milvus::GuaranteeStrongTs());

    std::uniform_int_distribution<int64_t> int64_gen(0, row_count - 1);
    int64_t q_number_1 = int64_gen(ran);
    int64_t q_number_2 = int64_gen(ran);
    s_arguments.AddTargetVector(field_face, std::move(insert_vectors[q_number_1]));
    s_arguments.AddTargetVector(field_face, std::move(insert_vectors[q_number_2]));
    std::cout << "Searching the No." << q_number_1 << " and No." << q_number_2 << " with expression: " << filter_expr
              << std::endl;

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

        auto name_field = result.OutputField(field_name);
        milvus::VarCharFieldDataPtr name_field_ptr = std::static_pointer_cast<milvus::VarCharFieldData>(name_field);
        auto& name_data = name_field_ptr->Data();

        auto age_field = result.OutputField(field_age);
        milvus::Int8FieldDataPtr age_field_ptr = std::static_pointer_cast<milvus::Int8FieldData>(age_field);
        auto& age_data = age_field_ptr->Data();

        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << "\tID: " << ids[i] << "\tDistance: " << distances[i] << "\tName: " << name_data[i]
                      << "\tAge: " << static_cast<int>(age_data[i]) << std::endl;
            // validate the age value
            if (insert_ages[ids[i]] != age_data[i]) {
                std::cout << "ERROR! The returned value doesn't match the inserted value" << std::endl;
            }
        }
    }

    // release collection
    status = client->ReleaseCollection(collection_name);
    CheckStatus("Failed to release collection:", status);
    std::cout << "Release collection " << collection_name << std::endl;

    // drop index
    status = client->DropIndex(collection_name, field_face);
    CheckStatus("Failed to drop index:", status);
    std::cout << "Drop index for field: " << field_face << std::endl;

    // drop partition
    status = client->DropPartition(collection_name, partition_name);
    CheckStatus("Failed to drop partition:", status);
    std::cout << "Drop partition " << partition_name << std::endl;

    // verify the row count should be 0
    // Note: call GetCollectionStatistics immediately after DropPartition could return non-zero value
    // wait a few seconds to get the correct zero value.
    std::this_thread::sleep_for(std::chrono::seconds(5));
    milvus::CollectionStat col_stat;
    status = client->GetCollectionStatistics(collection_name, col_stat);
    CheckStatus("Failed to get collection statistics:", status);
    std::cout << "Collection " << collection_name << " row count: " << col_stat.RowCount() << std::endl;

    // drop collection
    status = client->DropCollection(collection_name);
    CheckStatus("Failed to drop collection:", status);
    std::cout << "Drop collection " << collection_name << std::endl;

    client->Disconnect();
    return 0;
}
