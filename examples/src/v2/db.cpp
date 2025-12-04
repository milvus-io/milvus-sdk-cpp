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

    milvus::ListDatabasesResponse resp_list_dbs;
    status = client->ListDatabases(milvus::ListDatabasesRequest(), resp_list_dbs);
    util::CheckStatus("list databases", status);

    const std::string my_db_name = "my_temp_db_for_cpp_test";
    std::cout << "Databases: ";
    for (const auto& name : resp_list_dbs.DatabaseNames()) {
        std::cout << name << ",";
    }
    std::cout << std::endl;

    std::unordered_map<std::string, std::string> props;
    props.emplace("database.replica.number", "2");
    status = client->CreateDatabase(
        milvus::CreateDatabaseRequest().WithDatabaseName(my_db_name).WithProperties(std::move(props)));
    util::CheckStatus("create database: " + my_db_name, status);

    milvus::DescribeDatabaseResponse resp_desc_db;
    status = client->DescribeDatabase(milvus::DescribeDatabaseRequest().WithDatabaseName(my_db_name), resp_desc_db);
    util::CheckStatus("describe database: " + my_db_name, status);
    std::cout << "database.replica.number = " << resp_desc_db.Desc().Properties().at("database.replica.number")
              << std::endl;

    status = client->UseDatabase(my_db_name);
    util::CheckStatus("switch database:" + my_db_name, status);
    std::string current_db_name;
    client->CurrentUsedDatabase(current_db_name);
    std::cout << "Current in-used database: " << current_db_name << std::endl;

    // drop the collection if it exists
    const std::string collection_name = "CPP_V2_DB";
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));

    // create a collection
    const std::string field_id = "user_id";
    const std::string field_name = "user_name";
    const std::string field_age = "user_age";
    const std::string field_face = "user_face";
    const uint32_t dimension = 128;

    // collection schema, create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>(collection_name);
    collection_schema->AddField({field_id, milvus::DataType::INT64, "user id", true, false});
    milvus::FieldSchema varchar_scheam{field_name, milvus::DataType::VARCHAR, "user name"};
    varchar_scheam.SetMaxLength(100);
    collection_schema->AddField(varchar_scheam);
    collection_schema->AddField({field_age, milvus::DataType::INT8, "user age"});
    collection_schema->AddField(
        milvus::FieldSchema(field_face, milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(dimension));

    status = client->CreateCollection(milvus::CreateCollectionRequest().WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    // create indexes
    milvus::IndexDesc index_vector(field_face, "", milvus::IndexType::FLAT, milvus::MetricType::COSINE);
    milvus::IndexDesc index_varchar(field_name, "", milvus::IndexType::TRIE);
    milvus::IndexDesc index_sort(field_age, "", milvus::IndexType::STL_SORT);
    status = client->CreateIndex(milvus::CreateIndexRequest()
                                     .WithCollectionName(collection_name)
                                     .AddIndex(std::move(index_vector))
                                     .AddIndex(std::move(index_varchar))
                                     .AddIndex(std::move(index_sort)));
    util::CheckStatus("create indexes on collection", status);

    // create a partition
    std::string partition_name = "Year_2022";
    status = client->CreatePartition(
        milvus::CreatePartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name));
    util::CheckStatus("create partition: " + partition_name, status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + collection_name, status);

    // insert data by column-based
    const int64_t row_count = 1000;
    std::vector<int64_t> insert_ids;
    std::vector<std::string> insert_names;
    std::vector<int8_t> insert_ages;
    std::vector<std::vector<float>> insert_vectors = util::GenerateFloatVectors(dimension, row_count);
    for (auto i = 0; i < row_count; ++i) {
        insert_ids.push_back(i);
        insert_names.push_back("user_" + std::to_string(i));
        insert_ages.push_back(static_cast<int8_t>(util::RandomeValue<int>(1, 100)));
    }

    std::vector<milvus::FieldDataPtr> fields_data{
        std::make_shared<milvus::Int64FieldData>(field_id, insert_ids),
        std::make_shared<milvus::VarCharFieldData>(field_name, insert_names),
        std::make_shared<milvus::Int8FieldData>(field_age, insert_ages),
        std::make_shared<milvus::FloatVecFieldData>(field_face, insert_vectors)};
    milvus::InsertResponse resp_insert;
    status = client->Insert(milvus::InsertRequest()
                                .WithCollectionName(collection_name)
                                .WithPartitionName(partition_name)
                                .WithColumnsData(std::move(fields_data)),
                            resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted." << std::endl;

    // delete the item whose primary key is 5
    milvus::DeleteResponse resp_delete;
    status = client->Delete(milvus::DeleteRequest()
                                .WithCollectionName(collection_name)
                                .WithPartitionName(partition_name)
                                .WithFilter(field_id + "== 5"),
                            resp_delete);
    util::CheckStatus("delete entity whose id is 5", status);

    {
        // verify the row count of the partition is 999 by query(count(*))
        // set to STRONG level to ensure the delete request is done by server
        milvus::QueryRequest request;
        request.SetCollectionName(collection_name);
        request.AddPartitionName(partition_name);
        request.AddOutputField("count(*)");
        request.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query count(*) on partition", status);
        std::cout << "partition count(*) = " << response.Results().GetRowCount() << std::endl;
    }

    // now we switch back to the default database
    status = client->UseDatabase("default");
    util::CheckStatus("switch default database", status);
    client->CurrentUsedDatabase(current_db_name);
    std::cout << "Current in-used database: " << current_db_name << std::endl;

    {
        // query the deleted item and anthor item
        milvus::QueryRequest request;
        request.SetDatabaseName(my_db_name);  // we still can do search with our db name
        request.SetCollectionName(collection_name);
        request.AddPartitionName(partition_name);
        request.SetFilter(field_id + " in [5, 10]");
        request.AddOutputField(field_id);
        request.AddOutputField(field_name);
        request.AddOutputField(field_age);
        // set to EVENTUALLY level since the last query uses STRONG level and no data changed
        request.SetConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);

        std::cout << "Query with expression: " << request.Filter() << std::endl;
        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query", status);

        for (auto& field_data : response.Results().OutputFields()) {
            std::cout << "Field: " << field_data->Name() << " Count:" << field_data->Count() << std::endl;
        }
    }

    {
        // do search
        // this collection has only one vector field, no need to set the AnnsField name
        milvus::SearchRequest request;
        request.SetDatabaseName(my_db_name);  // we still can do search with our db name
        request.SetCollectionName(collection_name);
        request.AddPartitionName(partition_name);
        request.SetLimit(10);
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
        std::cout << "Searching the No." << q_number_1 << " and No." << q_number_2
                  << " with expression: " << filter_expr << std::endl;

        milvus::SearchResponse response;
        status = client->Search(request, response);
        util::CheckStatus("search", status);

        for (auto& result : response.Results().Results()) {
            auto ids = result.Ids().IntIDArray();
            auto distances = result.Scores();
            if (ids.size() != distances.size()) {
                std::cout << "Illegal result!" << std::endl;
                continue;
            }

            std::cout << "Result of one target vector:" << std::endl;

            auto name_field = result.OutputField<milvus::VarCharFieldData>(field_name);
            auto age_field = result.OutputField<milvus::Int8FieldData>(field_age);
            for (size_t i = 0; i < ids.size(); ++i) {
                std::cout << "\t" << result.PrimaryKeyName() << ":" << ids[i] << "\tDistance: " << distances[i] << "\t"
                          << name_field->Name() << ":" << name_field->Value(i) << "\t" << age_field->Name() << ":"
                          << +(age_field->Value(i)) << std::endl;
                // validate the age value
                if (insert_ages.at(ids[i]) != age_field->Value(i)) {
                    std::cout << "ERROR! The returned value doesn't match the inserted value" << std::endl;
                }
            }
        }
    }

    // now we switch back to our database, since some interfaces no db_name parameter
    status = client->UseDatabase(my_db_name);
    util::CheckStatus("switch database: " + my_db_name, status);
    client->CurrentUsedDatabase(current_db_name);
    std::cout << "Current in-used database: " << current_db_name << std::endl;

    // release collection
    status = client->ReleaseCollection(milvus::ReleaseCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("release collection: " + collection_name, status);

    // drop index
    status =
        client->DropIndex(milvus::DropIndexRequest().WithCollectionName(collection_name).WithFieldName(field_face));
    util::CheckStatus("drop index on field: " + field_face, status);

    // drop partition
    status = client->DropPartition(
        milvus::DropPartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name));
    util::CheckStatus("drop partition: " + partition_name, status);

    {
        // verify the row count should be 0
        // since the collection is not loaded, query(count(*)) cannot work.
        // Note:
        // 1. GetCollectionStatistics() only returns row number of sealed segments, and deleted items are not counted.
        // 2. call GetCollectionStatistics immediately after DropPartition could return non-zero value,
        //    wait a few seconds to get the correct zero value.
        std::this_thread::sleep_for(std::chrono::seconds(5));
        milvus::GetCollectionStatsResponse response;
        status = client->GetCollectionStats(milvus::GetCollectionStatsRequest().WithCollectionName(collection_name),
                                            response);
        util::CheckStatus("get collection statistics: " + collection_name, status);
        std::cout << "Collection " << collection_name << " row count: " << response.Stats().RowCount() << std::endl;
    }

    // drop collection
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("drop collection: " + collection_name, status);

    // now we switch back to the default database, prepare to delete our empty database
    status = client->UseDatabase(my_db_name);
    util::CheckStatus("switch default database", status);
    client->CurrentUsedDatabase(current_db_name);
    std::cout << "Current in-used database: " << current_db_name << std::endl;

    status = client->DropDatabase(milvus::DropDatabaseRequest().WithDatabaseName(my_db_name));
    util::CheckStatus("drop database: " + my_db_name, status);

    client->Disconnect();
    return 0;
}
