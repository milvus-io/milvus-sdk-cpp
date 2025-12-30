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

    const std::string collection_name = "CPP_V2_SPARSE_VECTOR";
    const std::string field_id = "pk";
    const std::string field_vector = "sparse";
    const std::string field_text = "text";

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>(collection_name);
    collection_schema->AddField({field_id, milvus::DataType::INT64, "user id", true, false});
    collection_schema->AddField({field_vector, milvus::DataType::SPARSE_FLOAT_VECTOR});
    collection_schema->AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(1024));

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::SPARSE_INVERTED_INDEX, milvus::MetricType::IP);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + collection_name, status);

    {
        // insert some rows by column-based
        auto ids = std::vector<int64_t>{10000, 10001};
        auto texts = std::vector<std::string>{"column-based-1", "column-based-2"};
        auto vectors = util::GenerateSparseVectors(100, 2);
        std::vector<milvus::FieldDataPtr> fields_data{
            std::make_shared<milvus::Int64FieldData>(field_id, ids),
            std::make_shared<milvus::VarCharFieldData>(field_text, texts),
            std::make_shared<milvus::SparseFloatVecFieldData>(field_vector, vectors)};

        milvus::InsertResponse resp_insert;
        status = client->Insert(
            milvus::InsertRequest().WithCollectionName(collection_name).WithColumnsData(std::move(fields_data)),
            resp_insert);
        util::CheckStatus("insert", status);
        std::cout << resp_insert.Results().InsertCount() << " rows inserted by column-based." << std::endl;
    }

    const int64_t row_count = 10;
    milvus::EntityRows rows;
    {
        // insert some rows by row-based
        for (auto i = 0; i < row_count; ++i) {
            milvus::EntityRow row;
            row[field_id] = i;
            row[field_text] = "this is text_" + std::to_string(i);
            row[field_vector] = util::GenerateSparseVectorInJson(100, true);
            rows.emplace_back(std::move(row));
        }

        milvus::InsertResponse resp_insert;
        milvus::EntityRows rows_copy = rows;  // the rows are used for search later, make a copy here
        status = client->Insert(
            milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows_copy)),
            resp_insert);
        util::CheckStatus("insert", status);
        std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;
    }

    {
        // query
        auto request =
            milvus::QueryRequest()
                .WithCollectionName(collection_name)
                .AddOutputField(field_vector)
                .AddOutputField(field_text)
                .WithLimit(5)
                // set to strong level so that the query is executed after the inserted data is consumed by server
                .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

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
        // do search
        auto request = milvus::SearchRequest()
                           .WithCollectionName(collection_name)
                           .WithLimit(3)
                           .WithAnnsField(field_vector)
                           .AddOutputField(field_vector)
                           .AddOutputField(field_text)
                           .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        auto q_number_1 = util::RandomeValue<int64_t>(0, row_count - 1);
        auto q_number_2 = util::RandomeValue<int64_t>(0, row_count - 1);
        auto q_vector_1 = rows[q_number_1][field_vector];
        auto q_vector_2 = rows[q_number_2][field_vector];
        request.AddSparseVector(q_vector_1);
        request.AddSparseVector(q_vector_2);

        std::cout << "Searching the ID." << q_number_1 << " sparse vector: " << q_vector_1 << std::endl;
        std::cout << "Searching the ID." << q_number_2 << " sparse vector: " << q_vector_2 << std::endl;

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
