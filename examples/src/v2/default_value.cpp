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

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    const std::string collection_name = "CPP_V2_DEFAULT_VALUE";
    const std::string partition_1 = "partition_1";
    const std::string partition_2 = "partition_2";
    const std::string field_id = "pk";
    const std::string field_vector = "vector";
    const std::string field_name = "name";
    const std::string field_price = "price";
    const uint32_t dimension = 4;

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->SetEnableDynamicField(true);
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, false});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema->AddField(
        milvus::FieldSchema(field_name, milvus::DataType::VARCHAR).WithMaxLength(1024).WithDefaultValue("No Name"));
    collection_schema->AddField(milvus::FieldSchema(field_price, milvus::DataType::FLOAT).WithDefaultValue(0.123456));

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    // create two partitions
    status = client->CreatePartition(
        milvus::CreatePartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_1));
    util::CheckStatus("create partition: " + partition_1, status);

    status = client->CreatePartition(
        milvus::CreatePartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_2));
    util::CheckStatus("create partition: " + partition_2, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::HNSW, milvus::MetricType::L2);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + collection_name, status);

    const int64_t row_count = 10;
    // insert 10 rows with default value by row-based
    {
        milvus::EntityRows rows;
        for (auto i = 0; i < row_count; ++i) {
            milvus::EntityRow row;
            row[field_id] = i;
            row[field_vector] = util::GenerateFloatVector(dimension);
            if (i % 2 == 0) {
                row[field_name] = "row_" + std::to_string(i);
                row[field_price] = static_cast<float>(i) / 4;
            } else {
                // the field_name and field_price are not set, recognized as default value
            }
            rows.emplace_back(std::move(row));
        }

        // insert into partition_1
        milvus::InsertResponse resp_insert;
        status = client->Insert(milvus::InsertRequest()
                                    .WithCollectionName(collection_name)
                                    .WithPartitionName(partition_1)
                                    .WithRowsData(std::move(rows)),
                                resp_insert);
        util::CheckStatus("insert", status);
        std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;
    }

    // insert 10 rows with null value by column-based
    {
        milvus::Int64FieldDataPtr id_field = std::make_shared<milvus::Int64FieldData>(field_id);
        milvus::FloatVecFieldDataPtr vector_field = std::make_shared<milvus::FloatVecFieldData>(field_vector);
        milvus::VarCharFieldDataPtr name_field = std::make_shared<milvus::VarCharFieldData>(field_name);

        for (auto i = 0; i < row_count; ++i) {
            id_field->Add(row_count + i);
            vector_field->Add(std::move(util::GenerateFloatVector(dimension)));
            name_field->Add("column_" + std::to_string(i));
            // the field_price is not provided, recognized as default value
        }

        // insert into partition_2
        std::vector<milvus::FieldDataPtr> fields_data{id_field, vector_field, name_field};
        milvus::InsertResponse resp_insert;
        status = client->Insert(milvus::InsertRequest()
                                    .WithCollectionName(collection_name)
                                    .WithPartitionName(partition_2)
                                    .WithColumnsData(std::move(fields_data)),
                                resp_insert);
        util::CheckStatus("insert", status);
        std::cout << resp_insert.Results().InsertCount() << " rows inserted by column-based." << std::endl;
    }

    {
        // query
        auto request =
            milvus::QueryRequest()
                .WithCollectionName(collection_name)
                .AddPartitionName(partition_1)
                .AddOutputField("*")
                .WithFilter(field_price + " < 0.5")  // query entity whose name is null value
                // set to strong level so that the query is executed after the inserted data is consumed by server
                .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        std::cout << "\nQuery with filter: " << request.Filter() << " in " << partition_1 << std::endl;
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
        // search
        auto request = milvus::SearchRequest()
                           .WithCollectionName(collection_name)
                           .WithFilter(field_name + " != \"No Name\"")  // search entitiies whose age is not null value
                           .WithLimit(20)
                           .WithAnnsField(field_vector)
                           .AddOutputField(field_name)
                           .AddOutputField(field_price)
                           .AddFloatVector(util::GenerateFloatVector(dimension))
                           .AddFloatVector(util::GenerateFloatVector(dimension))
                           .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        std::cout << "\nSearch with filter: " << request.Filter() << std::endl;
        milvus::SearchResponse response;
        status = client->Search(request, response);
        util::CheckStatus("search", status);

        for (auto& result : response.Results().Results()) {
            std::cout << "Result of one target vector:" << std::endl;
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
