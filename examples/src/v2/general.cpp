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
#include "milvus/MilvusClientV2.h"

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    milvus::CheckHealthResponse resp_health;
    status = client->CheckHealth(milvus::CheckHealthRequest(), resp_health);
    util::CheckStatus("check milvus server healthy", status);
    if (resp_health.IsHealthy()) {
        std::cout << "The milvus server is healthy" << std::endl;
    } else {
        std::cout << "The milvus server is unhealthy, reasons: " << std::endl;
        if (!resp_health.Reasons().empty()) {
            util::PrintList(resp_health.Reasons());
        }
        if (!resp_health.QuotaStates().empty()) {
            util::PrintList(resp_health.QuotaStates());
        }
    }

    // set timeout value for each rpc call
    client->SetRpcDeadlineMs(1000);

    // print the server version
    std::string version;
    status = client->GetServerVersion(version);
    util::CheckStatus("get server version", status);
    std::cout << "The milvus server version is: " << version << std::endl;

    // print the SDK version
    client->GetSDKVersion(version);
    std::cout << "The CPP SDK version is: " << version << std::endl;

    const std::string db_name = "cpp_sdk_test_db";
    const std::string collection_name = "CPP_V2_GENERAL";
    const std::string field_id = "user_id";
    const std::string field_name = "user_name";
    const std::string field_age = "user_age";
    const std::string field_face = "user_face";
    const uint32_t dimension = 128;

    // create database
    milvus::ListDatabasesResponse resp_list_dbs;
    status = client->ListDatabases(milvus::ListDatabasesRequest(), resp_list_dbs);
    util::CheckStatus("list databases", status);
    auto it = std::find(resp_list_dbs.DatabaseNames().begin(), resp_list_dbs.DatabaseNames().end(), db_name);
    if (it == resp_list_dbs.DatabaseNames().end()) {
        status = client->CreateDatabase(milvus::CreateDatabaseRequest().WithDatabaseName(db_name));
        util::CheckStatus("create database: " + db_name, status);
    }

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>(collection_name);
    collection_schema->AddField({field_id, milvus::DataType::INT64, "user id", true, false});
    milvus::FieldSchema varchar_scheam{field_name, milvus::DataType::VARCHAR, "user name"};
    varchar_scheam.SetMaxLength(100);
    collection_schema->AddField(varchar_scheam);
    collection_schema->AddField({field_age, milvus::DataType::INT8, "user age"});
    collection_schema->AddField(
        milvus::FieldSchema(field_face, milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(dimension));

    status = client->DropCollection(
        milvus::DropCollectionRequest().WithCollectionName(collection_name).WithDatabaseName(db_name));
    status = client->CreateCollection(milvus::CreateCollectionRequest()
                                          .WithCollectionSchema(collection_schema)
                                          .WithDatabaseName(db_name)
                                          .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG));
    util::CheckStatus("create collection: " + collection_name + " in database: " + db_name, status);

    // create index
    milvus::IndexDesc index_vector(field_face, "", milvus::IndexType::IVF_FLAT, milvus::MetricType::COSINE);
    index_vector.AddExtraParam(milvus::NLIST, "100");
    milvus::IndexDesc index_sort(field_age, "", milvus::IndexType::STL_SORT);
    milvus::IndexDesc index_varchar(field_name, "", milvus::IndexType::TRIE);
    status = client->CreateIndex(milvus::CreateIndexRequest()
                                     .WithCollectionName(collection_name)
                                     .WithDatabaseName(db_name)
                                     .AddIndex(std::move(index_vector))
                                     .AddIndex(std::move(index_sort))
                                     .AddIndex(std::move(index_varchar)));
    util::CheckStatus("create indexes on collection", status);

    // create a partition
    std::string partition_name = "Year_2022";
    status = client->CreatePartition(milvus::CreatePartitionRequest()
                                         .WithDatabaseName(db_name)
                                         .WithCollectionName(collection_name)
                                         .WithPartitionName(partition_name));
    util::CheckStatus("create partition: " + partition_name, status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest()
                                        .WithDatabaseName(db_name)
                                        .WithCollectionName(collection_name)
                                        .WithReplicaNum(1));
    util::CheckStatus("load collection: " + collection_name, status);

    // list collections
    milvus::ListCollectionsResponse resp_list_coll;
    status = client->ListCollections(milvus::ListCollectionsRequest().WithDatabaseName(db_name), resp_list_coll);
    util::CheckStatus("list collections in database: " + db_name, status);
    std::cout << "\nCollections:" << std::endl;
    for (auto& name : resp_list_coll.CollectionNames()) {
        std::cout << "\t" << name << std::endl;
    }

    // list partitions of the collection
    milvus::ListPartitionsResponse resp_list_part;
    status = client->ListPartitions(
        milvus::ListPartitionsRequest().WithDatabaseName(db_name).WithCollectionName(collection_name), resp_list_part);
    util::CheckStatus("list partitions", status);
    std::cout << "\nPartitions of " << collection_name << ":" << std::endl;
    for (auto& info : resp_list_part.PartitionInfos()) {
        std::cout << "\t" << info.Name() << std::endl;
    }

    // switch to the database
    status = client->UseDatabase(db_name);
    util::CheckStatus("use database: " + db_name, status);

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
        insert_vectors.emplace_back(std::move(util::GenerateFloatVector(dimension)));
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
        milvus::InsertResponse resp_insert;
        // since we have switched to db_name, we don't need to set DatabaseName here
        status = client->Insert(milvus::InsertRequest()
                                    .WithCollectionName(collection_name)
                                    .WithPartitionName(partition_name)
                                    .WithColumnsData(std::move(fields_data)),
                                resp_insert);
        util::CheckStatus("insert", status);
        std::cout << resp_insert.Results().InsertCount() << " rows inserted by column-based." << std::endl;
    }

    {
        // insert other rows by row-based
        milvus::EntityRows rows;
        for (auto i = column_based_count; i < insert_ids.size(); i++) {
            milvus::EntityRow row;
            row[field_id] = insert_ids[i];
            row[field_name] = insert_names[i];
            row[field_age] = insert_ages[i];
            row[field_face] = insert_vectors[i];
            rows.emplace_back(std::move(row));

            // insert batch by batch, batch size is 80
            if (rows.size() >= 80 || (i >= insert_ids.size() - 1)) {
                milvus::InsertResponse resp_insert;
                // since we have switched to db_name, we don't need to set DatabaseName here
                status = client->Insert(milvus::InsertRequest()
                                            .WithCollectionName(collection_name)
                                            .WithPartitionName(partition_name)
                                            .WithRowsData(std::move(rows)),
                                        resp_insert);
                util::CheckStatus("insert", status);
                std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;
                rows.clear();
            }
        }
    }

    {
        // delete one item whose primary key is 5
        milvus::DeleteResponse resp_delete;
        status = client->Delete(milvus::DeleteRequest()
                                    .WithCollectionName(collection_name)
                                    .WithPartitionName(partition_name)
                                    .WithFilter(field_id + "== 5"),
                                resp_delete);
        util::CheckStatus("delete entity whose id is 5", status);
    }

    {
        // verify the row count of the partition is 999 by query(count(*))
        // the collection default level is set to STRONG level, no need to set consistency level here
        milvus::QueryRequest request;
        request.SetCollectionName(collection_name);
        request.AddPartitionName(partition_name);
        request.AddOutputField("count(*)");

        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query count(*) on partition", status);
        std::cout << "partition count(*) = " << response.Results().GetRowCount() << std::endl;
    }

    {
        // query the deleted item and some other item, the returned result will not contain the deleted item
        milvus::QueryRequest request;
        request.SetCollectionName(collection_name);
        request.AddPartitionName(partition_name);
        request.SetFilter(field_id + " in [1, 5, 10]");
        request.AddOutputField(field_id);
        request.AddOutputField(field_name);
        request.AddOutputField(field_age);
        // set to EVENTUALLY level since the last query uses STRONG level and no data changed
        request.SetConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);

        std::cout << "\nQuery with expression: " << request.Filter() << std::endl;
        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query", status);

        auto query_results = response.Results();
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
        // the AnnsField name is passed by AddFloatVector()
        milvus::SearchRequest request;
        request.SetCollectionName(collection_name);
        request.AddPartitionName(partition_name);
        request.SetLimit(5);
        request.AddExtraParam(milvus::NPROBE, "10");
        request.AddOutputField(field_name);
        request.AddOutputField(field_age);
        std::string filter_expr = field_age + " > 40";
        request.SetFilter(filter_expr);
        // set to BOUNDED level to accept data inconsistence within a time window(default is 5 seconds)
        request.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        auto q_number_1 = util::RandomeValue<int64_t>(0, row_count - 1);
        auto q_number_2 = util::RandomeValue<int64_t>(0, row_count - 1);
        request.AddFloatVector(field_face, insert_vectors[q_number_1]);
        request.AddFloatVector(field_face, insert_vectors[q_number_2]);
        std::cout << "\nSearching the No." << q_number_1 << " and No." << q_number_2
                  << " with expression: " << filter_expr << std::endl;

        milvus::SearchResponse response;
        status = client->Search(request, response);
        util::CheckStatus("search", status);

        auto search_results = response.Results();
        for (auto& result : search_results.Results()) {
            std::cout << "Result of one target vector:" << std::endl;
            milvus::EntityRows output_rows;
            status = result.OutputRows(output_rows);
            util::CheckStatus("get output rows", status);
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
    status = client->ReleaseCollection(milvus::ReleaseCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("release collection: " + collection_name, status);

    // drop index
    status =
        client->DropIndex(milvus::DropIndexRequest().WithCollectionName(collection_name).WithFieldName(field_face));
    util::CheckStatus("drop index for field: " + field_face, status);

    // drop partition
    status = client->DropPartition(
        milvus::DropPartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name));
    util::CheckStatus("drop partition: " + partition_name, status);

    {
        // verify the row count should be 0
        // since the collection is not loaded, query(count(*)) cannot work.
        // Note:
        // 1. GetCollectionStatistics() only returns row number of sealed segments, and deleted items are not
        // 2. call GetCollectionStatistics immediately after DropPartition could return non-zero value,
        //    wait a few seconds to get the correct zero value.
        std::this_thread::sleep_for(std::chrono::seconds(5));
        milvus::GetCollectionStatsResponse response;
        status = client->GetCollectionStats(milvus::GetCollectionStatsRequest().WithCollectionName(collection_name),
                                            response);
        util::CheckStatus("get collection statistics", status);
        std::cout << "Collection " << collection_name << " row count: " << response.Stats().RowCount() << std::endl;
    }

    // drop collection
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("drop collection: " + collection_name, status);

    client->Disconnect();
    return 0;
}
