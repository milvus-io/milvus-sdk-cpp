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
    util::CheckStatus("connect milvus server", status);

    const std::string collection_name = "CPP_V1_PARTITION_KEY";
    const std::string field_id = "id";
    const std::string field_name = "name";
    const std::string field_vector = "vector";
    const uint32_t dimension = 128;

    // collection schema, drop and create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "", true, true});
    milvus::FieldSchema name_schema{field_name, milvus::DataType::VARCHAR, "partition key"};
    name_schema.SetMaxLength(100);
    name_schema.SetPartitionKey(true);  // set this field to be partition key
    collection_schema.AddField(name_schema);
    collection_schema.AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR, "embedding").WithDimension(dimension));

    status = client->DropCollection(collection_name);
    status = client->CreateCollection(collection_schema, 8);
    util::CheckStatus("create collection: " + collection_name, status);

    // create index (required after 2.2.0)
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::HNSW, milvus::MetricType::IP);
    index_vector.AddExtraParam("M", "64");
    index_vector.AddExtraParam("efConstruction", "100");
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("load collection: " + collection_name, status);

    // list partitions of the collection
    milvus::PartitionsInfo partitions_info;
    status = client->ListPartitions(collection_name, partitions_info);
    util::CheckStatus("list partitions", status);
    std::cout << "\nPartitions of " << collection_name << ":" << std::endl;
    for (auto& info : partitions_info) {
        std::cout << "\t" << info.Name() << std::endl;
    }

    // insert other rows by row-based
    // the data is split into different partitions according to the hash value of each partition key value
    // for example: "name_2_32" might be hashed into partition_1, "name_5_700" might be hashed into partition_4
    for (auto i = 0; i < 10; i++) {
        milvus::EntityRows rows;
        for (auto j = 0; j < 1000; j++) {
            milvus::EntityRow row;
            row[field_name] = "name_" + std::to_string(i) + "_" + std::to_string(j);
            row[field_vector] = util::GenerateFloatVector(dimension);
            rows.emplace_back(std::move(row));
        }

        milvus::DmlResults dml_results;
        status = client->Insert(collection_name, "", rows, dml_results);
        util::CheckStatus("insert", status);
        std::cout << dml_results.InsertCount() << " rows inserted by row-based." << std::endl;
    }

    {
        // verify the row count
        // set to STRONG level to ensure the delete request is done by server
        milvus::QueryArguments q_count{};
        q_count.SetCollectionName(collection_name);
        q_count.AddOutputField("count(*)");
        q_count.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::QueryResults count_result{};
        status = client->Query(q_count, count_result);
        util::CheckStatus("query count(*) on partition", status);
        std::cout << "partition count(*) = " << count_result.GetRowCount() << std::endl;
    }

    {
        // query with filter expression, the expression contains the partition key name
        // milvus only scans one partition, faster than scanning in entire collection
        milvus::QueryArguments q_arguments{};
        q_arguments.SetCollectionName(collection_name);
        q_arguments.SetFilter(field_name + " == \"name_3_500\"");
        q_arguments.AddOutputField(field_id);
        q_arguments.AddOutputField(field_name);
        // set to EVENTUALLY level since the last query uses STRONG level and no data changed
        q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);

        std::cout << "\nQuery with expression: " << q_arguments.Filter() << std::endl;
        milvus::QueryResults query_results{};
        status = client->Query(q_arguments, query_results);
        util::CheckStatus("query", status);

        milvus::EntityRows output_rows;
        status = query_results.OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        std::cout << "Query results:" << std::endl;
        for (const auto& row : output_rows) {
            std::cout << "\t" << row << std::endl;
        }
    }

    {
        // query with filter expression, the expression contains the partition key name
        // milvus only search in one partition, faster than searching in entire collection
        milvus::SearchArguments s_arguments{};
        s_arguments.SetCollectionName(collection_name);
        s_arguments.SetLimit(5);
        s_arguments.AddExtraParam("ef", "10");
        s_arguments.AddOutputField(field_id);
        s_arguments.AddOutputField(field_name);
        s_arguments.SetFilter(field_name + " == \"name_3_500\"");
        // set to BOUNDED level to accept data inconsistence within a time window(default is 5 seconds)
        s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        s_arguments.AddFloatVector(field_vector, util::GenerateFloatVector(dimension));
        std::cout << "\nSearching with expression: " << s_arguments.Filter() << std::endl;

        milvus::SearchResults search_results{};
        status = client->Search(s_arguments, search_results);
        util::CheckStatus("search", status);

        for (auto& result : search_results.Results()) {
            milvus::EntityRows output_rows;
            status = result.OutputRows(output_rows);
            util::CheckStatus("get output rows", status);
            for (const auto& row : output_rows) {
                std::cout << "\t" << row << std::endl;
            }
        }
    }

    client->Disconnect();
    return 0;
}
