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

    const std::string collection_name = "CPP_V1_DYNAMIC_FIELD";
    const std::string field_id = "pk";
    const std::string field_vector = "vector";
    const std::string field_text = "text";
    const uint32_t dimension = 4;

    // collection schema, drop and create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.SetEnableDynamicField(true);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "user id", true, false});
    collection_schema.AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema.AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(1024));

    status = client->DropCollection(collection_name);
    status = client->CreateCollection(collection_schema);
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::IVF_SQ8, milvus::MetricType::IP);
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("load collection: " + collection_name, status);

    const int64_t row_count = 10;
    {
        // insert 10 rows by column-based
        std::vector<int64_t> ids;
        std::vector<std::string> texts;
        std::vector<std::vector<float>> vectors;
        std::vector<nlohmann::json> dynamics;  //  a special JSON list to store dynamic fields
        for (auto i = 0; i < row_count; ++i) {
            ids.push_back(i);  // id from 0 to 9
            texts.push_back("text_" + std::to_string(i));
            vectors.emplace_back(std::move(util::GenerateFloatVector(dimension)));

            // add dynamic filed "a" and "b"
            nlohmann::json dynamic_data;
            dynamic_data["a"] = i;  // dynamic "a", values from 0 to 9
            if (i % 2 == 0) {
                dynamic_data["b"] = "column-based insert value is " + std::to_string(i);
            }
            dynamics.emplace_back(dynamic_data);
        }

        // milvus::DYNAMIC_FIELD is name of a special JSON field to store dynamic fileds in milvus
        // for column-based insert, we only support inserting dynamic fields in this way
        std::vector<milvus::FieldDataPtr> fields_data{
            std::make_shared<milvus::Int64FieldData>(field_id, ids),
            std::make_shared<milvus::VarCharFieldData>(field_text, texts),
            std::make_shared<milvus::FloatVecFieldData>(field_vector, vectors),
            std::make_shared<milvus::JSONFieldData>(milvus::DYNAMIC_FIELD, dynamics)};
        milvus::DmlResults dml_results;
        status = client->Insert(collection_name, "", fields_data, dml_results);
        util::CheckStatus("insert", status);
        std::cout << dml_results.InsertCount() << " rows inserted by column-based." << std::endl;
    }

    milvus::EntityRows rows;
    {
        // insert 10 rows by row-based
        for (auto i = 0; i < row_count; ++i) {
            milvus::EntityRow row;
            row[field_id] = row_count + i;  // id from 10 to 19
            row[field_text] = "this is text_" + std::to_string(i);
            row[field_vector] = util::GenerateFloatVector(dimension);
            row["a"] = row_count + i;  // dynamic "a", values from 10 to 19
            row["b"] = "row-based insert value is " + std::to_string(row_count + i);
            rows.emplace_back(std::move(row));
        }

        milvus::DmlResults dml_results;
        status = client->Insert(collection_name, "", rows, dml_results);
        util::CheckStatus("insert", status);
        std::cout << dml_results.InsertCount() << " rows inserted by row-based." << std::endl;
    }

    // query
    milvus::QueryArguments q_arguments{};
    q_arguments.SetCollectionName(collection_name);
    q_arguments.AddOutputField("*");
    q_arguments.SetFilter(field_id + " == 2");
    // set to strong level so that the query is executed after the inserted data is consumed by server
    q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

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

    // do search
    milvus::SearchArguments s_arguments{};
    s_arguments.SetCollectionName(collection_name);
    s_arguments.SetFilter("a in [4, 7, 13, 18]");  // filter on dynamic field
    s_arguments.SetLimit(10);
    s_arguments.AddOutputField(field_text);
    s_arguments.AddOutputField("a");
    s_arguments.AddOutputField("b");
    s_arguments.AddFloatVector(field_vector, util::GenerateFloatVector(dimension));
    s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

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

    client->Disconnect();
    return 0;
}
