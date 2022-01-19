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

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530};
    auto status = client->Connect(connect_param);
    if (!status.IsOk()) {
        std::cout << "Failed to connect milvus server: " << status.Message() << std::endl;
        return 0;
    }

    std::cout << "Connect to milvus server." << std::endl;

    // drop the collection if it exists
    std::string collection_name = "aaa";
    status = client->DropCollection(collection_name);

    // create a collection
    const uint32_t dimension = 4;
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({"identity", milvus::DataType::INT64, "user id", true, true});
    collection_schema.AddField({"age", milvus::DataType::INT8, "user age"});
    collection_schema.AddField(
        milvus::FieldSchema("face", milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(dimension));

    status = client->CreateCollection(collection_schema);
    if (!status.IsOk()) {
        std::cout << "Failed to create collection: " << collection_name << " error: " << status.Message() << std::endl;
        return 0;
    }

    std::cout << "Successfully create collection." << std::endl;

    // create a partition
    std::string partition_name = "Year_2022";
    status = client->CreatePartition(collection_name, partition_name);
    if (!status.IsOk()) {
        std::cout << "Failed to create partition: " << partition_name << " error: " << status.Message() << std::endl;
        return 0;
    }

    std::cout << "Successfully create partition." << std::endl;

    // tell server prepare to load collection
    milvus::ProgressMonitor pm = milvus::ProgressMonitor::NoWait();
    status = client->LoadCollection(collection_name, pm);

    // insert some rows
    const int32_t row_count = 100;
    std::vector<int16_t> ages;
    std::vector<std::vector<float>> vectors;
    std::default_random_engine ran;
    std::uniform_int_distribution<int16_t> int_gen(1, 100);
    std::uniform_real_distribution<float> float_gen(0.0, 1.0);
    for (auto i = 0; i < row_count; ++i) {
        ages.push_back(int_gen(ran));
        std::vector<float> vector(dimension);

        for (auto i = 0; i < dimension; ++i) {
            vector[i] = float_gen(ran);
        }
        vectors.emplace_back(vector);
    }

    std::vector<milvus::FieldDataPtr> fields_data{std::make_shared<milvus::Int16FieldData>("age", ages),
                                                  std::make_shared<milvus::FloatVecFieldData>("face", vectors)};
    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, partition_name, fields_data, dml_results);
    if (!status.IsOk()) {
        std::cout << "Failed to insert: " << partition_name << " error: " << status.Message() << std::endl;
        return 0;
    }

    std::cout << "Successfully insert " << dml_results.IdArray().IntIDArray().size() << " rows." << std::endl;

    // do search
    milvus::SearchArguments arguments{};
    arguments.SetCollectionName(collection_name);
    arguments.AddPartitionName(partition_name);
    arguments.SetTopK(10);
    arguments.AddOutputField("age");
    arguments.SetExpression("age > 1");
    // set to strong guarantee so that the search is executed after the inserted data is persisted
    arguments.SetGuaranteeTimestamp(milvus::GuaranteeStrongTs());
    int32_t q_number = row_count - 1;
    std::vector<float> q_vector = vectors[q_number];
    arguments.AddTargetVector("face", q_vector);
    milvus::SearchResults search_results{};
    status = client->Search(arguments, search_results);
    if (!status.IsOk()) {
        std::cout << "Failed to search, error: " << status.Message() << std::endl;
        return 0;
    }

    std::cout << "Successfully search." << std::endl;

    // TODO: here return empty results, not sure why, need more investigation
    auto& results = search_results.Results();
    std::cout << "Topk IDs: ";
    for (auto id : results[0].Ids().IntIDArray()) {
        std::cout << id << ",";
    }
    std::cout << std::endl;

    std::cout << "Topk distances: ";
    for (auto score : results[0].Scores()) {
        std::cout << score << ",";
    }
    std::cout << std::endl;

    status = client->DropCollection(collection_name);
    std::cout << "Drop collection " << collection_name << std::endl;

    printf("Example stop...\n");
    return 0;
}
