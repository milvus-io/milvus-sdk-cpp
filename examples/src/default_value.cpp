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

    const std::string collection_name = "TEST_CPP_DEFAULT_VALUE";
    const std::string partition_1 = "partition_1";
    const std::string partition_2 = "partition_2";
    const std::string field_id = "pk";
    const std::string field_vector = "vector";
    const std::string field_name = "name";
    const std::string field_price = "price";
    const uint32_t dimension = 4;

    // collection schema, drop and create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.SetEnableDynamicField(true);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "", true, false});
    collection_schema.AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema.AddField(
        milvus::FieldSchema(field_name, milvus::DataType::VARCHAR).WithMaxLength(1024).WithDefaultValue("No Name"));
    collection_schema.AddField(milvus::FieldSchema(field_price, milvus::DataType::FLOAT).WithDefaultValue(0.123456));

    status = client->DropCollection(collection_name);
    status = client->CreateCollection(collection_schema);
    util::CheckStatus("create collection: " + collection_name, status);

    // create two partitions
    status = client->CreatePartition(collection_name, partition_1);
    util::CheckStatus("create partition: " + partition_1, status);

    status = client->CreatePartition(collection_name, partition_2);
    util::CheckStatus("create partition: " + partition_2, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::HNSW, milvus::MetricType::L2);
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
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
        milvus::DmlResults dml_results;
        status = client->Insert(collection_name, partition_1, rows, dml_results);
        util::CheckStatus("insert", status);
        std::cout << dml_results.InsertCount() << " rows inserted by row-based." << std::endl;
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
        milvus::DmlResults dml_results;
        status = client->Insert(collection_name, partition_2, fields_data, dml_results);
        util::CheckStatus("insert", status);
        std::cout << dml_results.InsertCount() << " rows inserted by column-based." << std::endl;
    }

    {
        // query
        milvus::QueryArguments q_arguments{};
        q_arguments.SetCollectionName(collection_name);
        q_arguments.AddPartitionName(partition_1);
        q_arguments.AddOutputField("*");
        q_arguments.SetFilter(field_price + " < 0.5");  // query entity whose name is null value
        // set to strong level so that the query is executed after the inserted data is consumed by server
        q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        std::cout << "\nQuery with filter: " << q_arguments.Filter() << " in " << partition_1 << std::endl;
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
        // search
        milvus::SearchArguments s_arguments{};
        s_arguments.SetCollectionName(collection_name);
        s_arguments.SetFilter(field_name + " != \"No Name\"");  // search entitiies whose age is not null value
        s_arguments.SetLimit(20);
        s_arguments.AddOutputField(field_name);
        s_arguments.AddOutputField(field_price);
        s_arguments.AddFloatVector(field_vector, util::GenerateFloatVector(dimension));
        s_arguments.AddFloatVector(field_vector, util::GenerateFloatVector(dimension));
        s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        std::cout << "\nSearch with filter: " << s_arguments.Filter() << std::endl;
        milvus::SearchResults search_results{};
        status = client->Search(s_arguments, search_results);
        util::CheckStatus("search", status);

        for (auto& result : search_results.Results()) {
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
