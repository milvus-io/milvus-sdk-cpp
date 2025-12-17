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

    const std::string collection_name = "CPP_V2_ADD_FIELD";
    const std::string field_id = "pk";
    const std::string field_vector = "vector";
    const uint32_t dimension = 4;

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>(collection_name);
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, true});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(milvus::CreateCollectionRequest().WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::HNSW, milvus::MetricType::L2);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + collection_name, status);

    const int64_t row_count = 10;
    // insert 10 rows by row-based
    {
        milvus::EntityRows rows;
        for (auto i = 0; i < row_count; ++i) {
            milvus::EntityRow row;
            row[field_vector] = util::GenerateFloatVector(dimension);
            rows.emplace_back(std::move(row));
        }

        // insert into partition_1
        milvus::InsertResponse resp_insert;
        status = client->Insert(
            milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), resp_insert);
        util::CheckStatus("insert", status);
        std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;
    }

    // add more fields to the existing collection
    // new fields must be nullable
    {
        milvus::FieldSchema new_field_1 = milvus::FieldSchema()
                                              .WithName("new_1")
                                              .WithDataType(milvus::DataType::VARCHAR)
                                              .WithMaxLength(64)
                                              .WithNullable(true)
                                              .WithDefaultValue("default text");
        auto status = client->AddCollectionField(
            milvus::AddCollectionFieldRequest().WithCollectionName(collection_name).WithField(std::move(new_field_1)));
        util::CheckStatus("add a new varchar field", status);

        milvus::FieldSchema new_field_2 = milvus::FieldSchema()
                                              .WithName("new_2")
                                              .WithDataType(milvus::DataType::ARRAY)
                                              .WithElementType(milvus::DataType::INT16)
                                              .WithMaxCapacity(10)
                                              .WithNullable(true);
        status = client->AddCollectionField(
            milvus::AddCollectionFieldRequest().WithCollectionName(collection_name).WithField(std::move(new_field_2)));
        util::CheckStatus("add a new array field", status);
    }

    // insert another 10 rows by row-based
    {
        milvus::EntityRows rows;
        for (auto i = 0; i < row_count; ++i) {
            milvus::EntityRow row;
            row[field_vector] = util::GenerateFloatVector(dimension);
            row["new_1"] = "inserted value " + std::to_string(i);
            row["new_2"] = util::RandomeValues<int16_t>(0, 10, i % 10 + 1);
            rows.emplace_back(std::move(row));
        }

        // insert into partition_1
        milvus::InsertResponse resp_insert;
        status = client->Insert(
            milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), resp_insert);
        util::CheckStatus("insert", status);
        std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;
    }

    {
        // verify the row count is 20
        auto request = milvus::QueryRequest()
                           .WithCollectionName(collection_name)
                           .AddOutputField("count(*)")
                           .WithConsistencyLevel(
                               milvus::ConsistencyLevel::STRONG);  // set to strong level so that the query is executed
                                                                   // after the inserted data is consumed by server

        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query count(*)", status);
        std::cout << "count(*) = " << response.Results().GetRowCount() << std::endl;
    }

    {
        // query the 10 rows are default values for the added field
        auto request = milvus::QueryRequest()
                           .WithCollectionName(collection_name)
                           .AddOutputField("*")
                           .WithFilter("new_1 == 'default text'");

        std::cout << "\nQuery with filter: " << request.Filter() << std::endl;
        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query", status);

        milvus::EntityRows output_rows;
        status = response.Results().OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        std::cout << "Query results:" << std::endl;
        for (const auto& row : output_rows) {
            std::cout << "\t" << row << std::endl;
        }
    }

    {
        // query the 10 rows inserted by Insert() the added field
        auto request = milvus::QueryRequest()
                           .WithCollectionName(collection_name)
                           .AddOutputField("*")
                           .WithFilter("ARRAY_LENGTH(new_2) > 0");

        std::cout << "\nQuery with filter: " << request.Filter() << std::endl;
        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query", status);

        milvus::EntityRows output_rows;
        status = response.Results().OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        std::cout << "Query results:" << std::endl;
        for (const auto& row : output_rows) {
            std::cout << "\t" << row << std::endl;
        }
    }

    client->Disconnect();
    return 0;
}
