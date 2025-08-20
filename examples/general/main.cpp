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

    // set timeout value for each rpc call
    client->SetRpcDeadlineMs(1000);

    // print the server version
    std::string version;
    status = client->GetVersion(version);
    util::CheckStatus("Failed to get server version:", status);
    std::cout << "The milvus server version is: " << version << std::endl;

    // drop the collection if it exists
    const std::string collection_name = "TEST_CPP_GENERAL";
    status = client->DropCollection(collection_name);

    // create a collection
    const std::string field_id = "user_id";
    const std::string field_name = "user_name";
    const std::string field_age = "user_age";
    const std::string field_face = "user_face";
    const uint32_t dimension = 128;

    // collection schema, create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "user id", true, false});
    milvus::FieldSchema varchar_scheam{field_name, milvus::DataType::VARCHAR, "user name"};
    varchar_scheam.SetMaxLength(100);
    collection_schema.AddField(varchar_scheam);
    collection_schema.AddField({field_age, milvus::DataType::INT8, "user age"});
    collection_schema.AddField(
        milvus::FieldSchema(field_face, milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(dimension));

    status = client->CreateCollection(collection_schema);
    util::CheckStatus("Failed to create collection:", status);
    std::cout << "Successfully create collection " << collection_name << std::endl;

    // create index (required after 2.2.0)
    milvus::IndexDesc index_vector(field_face, "", milvus::IndexType::IVF_FLAT, milvus::MetricType::COSINE);
    index_vector.AddExtraParam(milvus::NLIST, "100");
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("Failed to create index on vector field:", status);
    std::cout << "Successfully create index." << std::endl;

    milvus::IndexDesc index_varchar(field_name, "", milvus::IndexType::TRIE);
    status = client->CreateIndex(collection_name, index_varchar);
    util::CheckStatus("Failed to create index on varchar field:", status);
    std::cout << "Successfully create index." << std::endl;

    milvus::IndexDesc index_sort(field_age, "", milvus::IndexType::STL_SORT);
    status = client->CreateIndex(collection_name, index_sort);
    util::CheckStatus("Failed to create index on integer field:", status);
    std::cout << "Successfully create index." << std::endl;

    // create a partition
    std::string partition_name = "Year_2022";
    status = client->CreatePartition(collection_name, partition_name);
    util::CheckStatus("Failed to create partition:", status);
    std::cout << "Successfully create partition." << std::endl;

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("Failed to load collection:", status);

    // list collections
    milvus::CollectionsInfo collections_info;
    status = client->ListCollections(collections_info);
    util::CheckStatus("Failed to list collections:", status);
    std::cout << "\nCollections:" << std::endl;
    for (auto& info : collections_info) {
        std::cout << "\t" << info.Name() << std::endl;
    }

    // list partitions of the collection
    milvus::PartitionsInfo partitions_info;
    status = client->ListPartitions(collection_name, partitions_info);
    util::CheckStatus("Failed to list partitions:", status);
    std::cout << "\nPartitions of " << collection_name << ":" << std::endl;
    for (auto& info : partitions_info) {
        std::cout << "\t" << info.Name() << std::endl;
    }

    // prepare original data
    const int64_t row_count = 1000;
    std::vector<int64_t> insert_ids;
    std::vector<std::string> insert_names;
    std::vector<int8_t> insert_ages;
    std::vector<std::vector<float>> insert_vectors;
    for (auto i = 0; i < row_count; ++i) {
        insert_ids.push_back(i);
        insert_names.push_back("user_" + std::to_string(i));
        insert_ages.push_back(static_cast<int8_t>(util::RandomeValue<int>(1, 100)));
        insert_vectors.emplace_back(util::GenerateFloatVector(dimension));
    }

    const uint64_t column_based_count = 500;
    {
        // insert 500 rows by column-based
        const uint64_t count = column_based_count;
        auto ids = std::vector<int64_t>(insert_ids.begin(), insert_ids.begin() + count);
        auto names = std::vector<std::string>(insert_names.begin(), insert_names.begin() + count);
        auto ages = std::vector<int8_t>(insert_ages.begin(), insert_ages.begin() + count);
        auto faces = std::vector<std::vector<float>>(insert_vectors.begin(), insert_vectors.begin() + count);
        std::vector<milvus::FieldDataPtr> fields_data{std::make_shared<milvus::Int64FieldData>(field_id, ids),
                                                      std::make_shared<milvus::VarCharFieldData>(field_name, names),
                                                      std::make_shared<milvus::Int8FieldData>(field_age, ages),
                                                      std::make_shared<milvus::FloatVecFieldData>(field_face, faces)};
        milvus::DmlResults dml_results;
        status = client->Insert(collection_name, partition_name, fields_data, dml_results);
        util::CheckStatus("Failed to insert:", status);
        std::cout << "Successfully insert " << dml_results.InsertCount() << " rows by column-based." << std::endl;
    }

    {
        // insert other rows by row-based
        std::vector<nlohmann::json> rows;
        for (auto i = column_based_count; i < insert_ids.size(); i++) {
            nlohmann::json row;
            row[field_id] = insert_ids[i];
            row[field_name] = insert_names[i];
            row[field_age] = insert_ages[i];
            row[field_face] = insert_vectors[i];
            rows.emplace_back(row);

            // insert batch by batch, batch size is 80
            if (rows.size() >= 80 || (i >= insert_ids.size() - 1)) {
                milvus::DmlResults dml_results;
                status = client->Insert(collection_name, partition_name, rows, dml_results);
                util::CheckStatus("Failed to insert:", status);
                std::cout << "Successfully insert " << dml_results.InsertCount() << " rows by row-based." << std::endl;
                rows.clear();
            }
        }
    }

    {
        // delete one item whose primary key is 5
        milvus::DmlResults del_res;
        status = client->Delete(collection_name, partition_name, field_id + "== 5", del_res);
        util::CheckStatus("Failed to delete entity:", status);
        std::cout << "Delete entity whose id is 5" << std::endl;
    }

    {
        // verify the row count of the partition is 999 by query(count(*))
        // set to STRONG level to ensure the delete request is done by server
        milvus::QueryArguments q_count{};
        q_count.SetCollectionName(collection_name);
        q_count.AddPartitionName(partition_name);
        q_count.AddOutputField("count(*)");
        q_count.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::QueryResults count_resutl{};
        status = client->Query(q_count, count_resutl);
        util::CheckStatus("Failed to query count(*):", status);
        std::cout << "Successfully query count(*) on partition." << std::endl;
        std::cout << "partition count(*) = " << count_resutl.GetRowCount() << std::endl;
    }

    {
        // query the deleted item and some other item, the returned result will not contain the deleted item
        milvus::QueryArguments q_arguments{};
        q_arguments.SetCollectionName(collection_name);
        q_arguments.AddPartitionName(partition_name);
        q_arguments.SetFilter(field_id + " in [1, 5, 10]");
        q_arguments.AddOutputField(field_id);
        q_arguments.AddOutputField(field_name);
        q_arguments.AddOutputField(field_age);
        // set to EVENTUALLY level since the last query uses STRONG level and no data changed
        q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);

        std::cout << "\nQuery with expression: " << q_arguments.Filter() << std::endl;
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
    }

    {
        // do search
        // this collection has only one vector field, no need to set the AnnsField name
        milvus::SearchArguments s_arguments{};
        s_arguments.SetCollectionName(collection_name);
        s_arguments.AddPartitionName(partition_name);
        s_arguments.SetLimit(5);
        s_arguments.AddExtraParam(milvus::NPROBE, "10");
        s_arguments.AddOutputField(field_name);
        s_arguments.AddOutputField(field_age);
        std::string filter_expr = field_age + " > 40";
        s_arguments.SetFilter(filter_expr);
        // set to BOUNDED level to accept data inconsistence within a time window(default is 5 seconds)
        s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        auto q_number_1 = util::RandomeValue<int64_t>(0, row_count - 1);
        auto q_number_2 = util::RandomeValue<int64_t>(0, row_count - 1);
        s_arguments.AddFloatVector(field_face, insert_vectors[q_number_1]);
        s_arguments.AddFloatVector(field_face, insert_vectors[q_number_2]);
        std::cout << "\nSearching the No." << q_number_1 << " and No." << q_number_2
                  << " with expression: " << filter_expr << std::endl;

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
                // validate the age value
                if (insert_ages[row[field_id]] != row[field_age]) {
                    std::cout << "ERROR! The returned value doesn't match the inserted value" << std::endl;
                }
            }
        }
    }

    // release collection
    status = client->ReleaseCollection(collection_name);
    util::CheckStatus("Failed to release collection:", status);
    std::cout << "Release collection " << collection_name << std::endl;

    // drop index
    status = client->DropIndex(collection_name, field_face);
    util::CheckStatus("Failed to drop index:", status);
    std::cout << "Drop index for field: " << field_face << std::endl;

    // drop partition
    status = client->DropPartition(collection_name, partition_name);
    util::CheckStatus("Failed to drop partition:", status);
    std::cout << "Drop partition " << partition_name << std::endl;

    {
        // verify the row count should be 0
        // since the collection is not loaded, query(count(*)) cannot work.
        // Note:
        // 1. GetCollectionStatistics() only returns row number of sealed segments, and deleted items are not counted.
        // 2. call GetCollectionStatistics immediately after DropPartition could return non-zero value,
        //    wait a few seconds to get the correct zero value.
        std::this_thread::sleep_for(std::chrono::seconds(5));
        milvus::CollectionStat col_stat;
        status = client->GetCollectionStatistics(collection_name, col_stat);
        util::CheckStatus("Failed to get collection statistics:", status);
        std::cout << "Collection " << collection_name << " row count: " << col_stat.RowCount() << std::endl;
    }

    // drop collection
    status = client->DropCollection(collection_name);
    util::CheckStatus("Failed to drop collection:", status);
    std::cout << "Drop collection " << collection_name << std::endl;

    client->Disconnect();
    return 0;
}
