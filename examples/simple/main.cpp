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
    const std::string field_id_name = "identity";
    const std::string field_age_name = "age";
    const std::string field_face_name = "face";
    const uint32_t dimension = 4;
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id_name, milvus::DataType::INT64, "user id", true, false});
    collection_schema.AddField({field_age_name, milvus::DataType::INT8, "user age"});
    collection_schema.AddField(milvus::FieldSchema(field_face_name, milvus::DataType::FLOAT_VECTOR, "face signature")
                                   .WithDimension(dimension));

    status = client->CreateCollection(collection_schema);
    CheckStatus("Failed to create collection:", status);
    std::cout << "Successfully create collection." << std::endl;

    // create index (required after 2.2.0)
    milvus::IndexDesc index_desc(field_face_name, "", milvus::IndexType::FLAT, milvus::MetricType::L2, 0);
    status = client->CreateIndex(collection_name, index_desc);
    CheckStatus("Failed to create index:", status);
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
    std::vector<int8_t> insert_ages;
    std::vector<std::vector<float>> insert_vectors;
    std::default_random_engine ran(time(nullptr));
    std::uniform_int_distribution<int> int_gen(1, 100);
    std::uniform_real_distribution<float> float_gen(0.0, 1.0);
    for (auto i = 0; i < row_count; ++i) {
        insert_ids.push_back(i);
        insert_ages.push_back(int_gen(ran));
        std::vector<float> vector(dimension);

        for (auto i = 0; i < dimension; ++i) {
            vector[i] = float_gen(ran);
        }
        insert_vectors.emplace_back(vector);
    }

    std::vector<milvus::FieldDataPtr> fields_data{
        std::make_shared<milvus::Int64FieldData>(field_id_name, insert_ids),
        std::make_shared<milvus::Int8FieldData>(field_age_name, insert_ages),
        std::make_shared<milvus::FloatVecFieldData>(field_face_name, insert_vectors)};
    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, partition_name, fields_data, dml_results);
    CheckStatus("Failed to insert:", status);
    std::cout << "Successfully insert " << dml_results.IdArray().IntIDArray().size() << " rows." << std::endl;

    // get partition statistics
    milvus::PartitionStat part_stat;
    status = client->GetPartitionStatistics(collection_name, partition_name, part_stat);
    CheckStatus("Failed to get partition statistics:", status);
    std::cout << "Partition " << partition_name << " row count: " << part_stat.RowCount() << std::endl;

    // do search
    milvus::SearchArguments arguments{};
    arguments.SetCollectionName(collection_name);
    arguments.AddPartitionName(partition_name);
    arguments.SetTopK(10);
    arguments.AddOutputField(field_age_name);
    arguments.SetExpression(field_age_name + " > 40");
    // set to strong guarantee so that the search is executed after the inserted data is persisted
    arguments.SetGuaranteeTimestamp(milvus::GuaranteeStrongTs());

    std::uniform_int_distribution<int64_t> int64_gen(0, row_count - 1);
    int64_t q_number = int64_gen(ran);
    std::vector<float> q_vector = insert_vectors[q_number];
    arguments.AddTargetVector<milvus::FloatVecFieldData>(field_face_name, std::move(q_vector));
    std::cout << "Searching the No." << q_number << " entity..." << std::endl;

    milvus::SearchResults search_results{};
    status = client->Search(arguments, search_results);
    CheckStatus("Failed to search:", status);
    std::cout << "Successfully search." << std::endl;

    for (auto& result : search_results.Results()) {
        auto& ids = result.Ids().IntIDArray();
        auto& distances = result.Scores();
        if (ids.size() != distances.size()) {
            std::cout << "Illegal result!" << std::endl;
            continue;
        }

        auto age_field = result.OutputField(field_age_name);
        milvus::Int8FieldDataPtr age_field_ptr = std::static_pointer_cast<milvus::Int8FieldData>(age_field);
        auto& age_data = age_field_ptr->Data();

        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << "ID: " << ids[i] << "\tDistance: " << distances[i]
                      << "\tAge: " << static_cast<int32_t>(age_data[i]) << std::endl;
            // validate the age value
            if (insert_ages[ids[i]] != age_data[i]) {
                std::cout << "ERROR! The returned value doesn't match the inserted value" << std::endl;
            }
        }
    }

    // drop partition
    status = client->DropPartition(collection_name, partition_name);
    CheckStatus("Failed to drop partition:", status);
    std::cout << "Drop partition " << partition_name << std::endl;

    // verify the row count should be 0
    milvus::CollectionStat col_stat;
    status = client->GetCollectionStatistics(collection_name, col_stat);
    CheckStatus("Failed to get collection statistics:", status);
    std::cout << "Collection " << collection_name << " row count: " << col_stat.RowCount() << std::endl;

    // drop collection
    status = client->DropCollection(collection_name);
    CheckStatus("Failed to drop collection:", status);
    std::cout << "Drop collection " << collection_name << std::endl;

    return 0;
}
