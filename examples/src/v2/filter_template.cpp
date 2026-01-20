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

    const std::string collection_name = "CPP_V2_FILTER_TEMPLATE";
    const std::string field_id = "pk";
    const std::string field_vector = "vector";
    const std::string field_text = "text";
    const uint32_t dimension = 4;

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField(milvus::FieldSchema(field_id, milvus::DataType::INT64, "", true, true));
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema->AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(1024));

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + collection_name, status);

    // insert some rows
    milvus::EntityRows rows;
    for (auto i = 0; i < 10000; ++i) {
        milvus::EntityRow row;  // id is auto-generated
        row[field_text] = "text_" + std::to_string(i);
        row[field_vector] = util::GenerateFloatVector(dimension);
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse resp_insert;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;
    auto ids = resp_insert.Results().IdArray().IntIDArray();

    {
        // query with filter template
        std::string filter = field_id + " in {my_ids}";  // "my_ids" is an alias will be used in filter template
        std::cout << "Query with filter expression: " << filter << std::endl;

        auto begin = ids.begin() + 500;
        auto end = begin + 100;
        std::vector<int64_t> filter_ids(begin, end);
        nlohmann::json filter_template = filter_ids;

        auto request =
            milvus::QueryRequest()
                .WithCollectionName(collection_name)
                .AddOutputField(field_text)
                .WithFilter(filter)
                .AddFilterTemplate("my_ids", filter_template)  // filter template
                // set to strong level so that the query is executed after the inserted data is consumed by server
                .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query", status);

        milvus::EntityRows output_rows;
        status = response.Results().OutputRows(output_rows);
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

        auto request = milvus::SearchRequest()
                           .WithCollectionName(collection_name)
                           .WithLimit(static_cast<int64_t>(texts.size()))
                           .WithFilter(filter)
                           .WithAnnsField(field_vector)
                           .AddFilterTemplate("my_texts", filter_template)
                           .AddOutputField(field_text)
                           .AddFloatVector(util::GenerateFloatVector(dimension))
                           .AddFloatVector(util::GenerateFloatVector(dimension))
                           .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        milvus::SearchResponse response;
        status = client->Search(request, response);
        util::CheckStatus("search", status);

        std::cout << "Search with filter template:" << std::endl;
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
