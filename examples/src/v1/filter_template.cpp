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

    const std::string collection_name = "CPP_V1_FILTER_TEMPLATE";
    const std::string field_id = "pk";
    const std::string field_vector = "vector";
    const std::string field_text = "text";
    const uint32_t dimension = 4;

    // collection schema, drop and create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField(milvus::FieldSchema(field_id, milvus::DataType::INT64, "", true, true));
    collection_schema.AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema.AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(1024));

    status = client->DropCollection(collection_name);
    status = client->CreateCollection(collection_schema);
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("load collection: " + collection_name, status);

    // insert some rows
    milvus::EntityRows rows;
    for (auto i = 0; i < 10000; ++i) {
        milvus::EntityRow row;  // id is auto-generated
        row[field_text] = "text_" + std::to_string(i);
        row[field_vector] = util::GenerateFloatVector(dimension);
        rows.emplace_back(std::move(row));
    }

    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", rows, dml_results);
    util::CheckStatus("insert", status);
    std::cout << dml_results.InsertCount() << " rows inserted by row-based." << std::endl;
    auto ids = dml_results.IdArray().IntIDArray();

    {
        // query with filter template
        std::string filter = field_id + " in {my_ids}";  // "my_ids" is an alias will be used in filter template
        std::cout << "Query with filter expression: " << filter << std::endl;

        auto begin = ids.begin() + 500;
        auto end = begin + 100;
        std::vector<int64_t> filter_ids(begin, end);
        nlohmann::json filter_template = filter_ids;

        milvus::QueryArguments q_arguments{};
        q_arguments.SetCollectionName(collection_name);
        q_arguments.AddOutputField(field_text);
        q_arguments.SetFilter(filter);
        q_arguments.AddFilterTemplate("my_ids", filter_template);  // filter template
        // set to strong level so that the query is executed after the inserted data is consumed by server
        q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::QueryResults query_results{};
        status = client->Query(q_arguments, query_results);
        util::CheckStatus("query", status);

        milvus::EntityRows output_rows;
        status = query_results.OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        std::cout << "Query with filter template:" << std::endl;
        for (const auto& row : output_rows) {
            std::cout << "\t" << row << std::endl;
        }
    }

    {
        // search with filter template
        std::string filter = field_text + " in {my_texts}";  // "my_texts" is an alias will be used in filter template
        std::vector<std::string> texts;
        for (auto i = 300; i < 500; i++) {
            texts.push_back("text_" + std::to_string(i));
        }
        nlohmann::json filter_template = texts;

        milvus::SearchArguments s_arguments{};
        s_arguments.SetCollectionName(collection_name);
        s_arguments.SetLimit(static_cast<int64_t>(texts.size()));
        s_arguments.SetFilter(filter);
        s_arguments.AddFilterTemplate("my_texts", filter_template);
        s_arguments.AddOutputField(field_text);
        s_arguments.AddFloatVector(field_vector, util::GenerateFloatVector(dimension));
        s_arguments.AddFloatVector(field_vector, util::GenerateFloatVector(dimension));
        s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        milvus::SearchResults search_results{};
        status = client->Search(s_arguments, search_results);
        util::CheckStatus("search", status);

        std::cout << "Search with filter template:" << std::endl;
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
