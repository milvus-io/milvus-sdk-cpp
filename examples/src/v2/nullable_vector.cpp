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

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {

constexpr uint32_t DIMENSION = 8;

void
InsertNullVectors(const milvus::MilvusClientV2Ptr& client) {
    const std::string collection_name = "CPP_V2_INSERT_NULL_VECTORS";
    const std::string field_id = "id";
    const std::string field_name = "name";
    const std::string field_vector = "embedding";

    std::cout << "=== Demo 1: Insert null vectors ===" << std::endl;
    client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));

    auto schema = std::make_shared<milvus::CollectionSchema>();
    schema->AddField(milvus::FieldSchema(field_id, milvus::DataType::INT64, "", true, false));
    schema->AddField(milvus::FieldSchema(field_name, milvus::DataType::VARCHAR).WithMaxLength(100));
    schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(DIMENSION).WithNullable(true));

    auto status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
    util::CheckStatus("create collection with nullable vector field", status);

    status = client->CreateIndex(
        milvus::CreateIndexRequest()
            .WithCollectionName(collection_name)
            .AddIndex(milvus::IndexDesc(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::L2)));
    util::CheckStatus("create index", status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection", status);

    constexpr int64_t row_count = 100;
    int64_t null_count = 0;
    milvus::EntityRows rows;
    rows.reserve(row_count);
    for (int64_t i = 0; i < row_count; ++i) {
        milvus::EntityRow row;
        row[field_id] = i;
        row[field_name] = "item_" + std::to_string(i);
        if (i % 2 == 0) {
            row[field_vector] = util::GenerateFloatVector(DIMENSION);
        } else {
            row[field_vector] = nullptr;
            ++null_count;
        }
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse insert_response;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            insert_response);
    util::CheckStatus("insert nullable vectors", status);
    std::cout << "Inserted " << insert_response.Results().InsertCount() << " rows: " << row_count - null_count
              << " valid, " << null_count << " null" << std::endl;

    milvus::QueryResponse query_response;
    status = client->Query(milvus::QueryRequest()
                               .WithCollectionName(collection_name)
                               .WithFilter(field_id + " >= 0")
                               .AddOutputField(field_id)
                               .AddOutputField(field_vector)
                               .WithLimit(row_count + 10)
                               .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                           query_response);
    util::CheckStatus("query nullable vectors", status);

    milvus::EntityRows output_rows;
    status = query_response.Results().OutputRows(output_rows);
    util::CheckStatus("get query output rows", status);
    int64_t query_null_count = 0;
    for (const auto& row : output_rows) {
        if (row.at(field_vector).is_null()) {
            ++query_null_count;
        }
    }
    std::cout << "Query result: " << output_rows.size() - query_null_count << " valid, " << query_null_count << " null"
              << std::endl;

    milvus::SearchResponse search_response;
    status = client->Search(milvus::SearchRequest()
                                .WithCollectionName(collection_name)
                                .WithAnnsField(field_vector)
                                .WithLimit(10)
                                .AddOutputField(field_id)
                                .AddOutputField(field_vector)
                                .AddFloatVector(util::GenerateFloatVector(DIMENSION))
                                .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                            search_response);
    util::CheckStatus("search nullable vectors", status);
    if (!search_response.Results().Results().empty()) {
        std::cout << "Search returned " << search_response.Results().Results().front().GetRowCount()
                  << " hits (only non-null vectors)" << std::endl;
    }

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("drop collection", status);
    std::cout << std::endl;
}

void
AddNullableVectorField(const milvus::MilvusClientV2Ptr& client) {
    const std::string collection_name = "CPP_V2_ADD_NULLABLE_VECTOR";
    const std::string field_id = "id";
    const std::string field_name = "name";
    const std::string field_vector_v1 = "embedding_v1";
    const std::string field_vector_v2 = "embedding_v2";

    std::cout << "=== Demo 2: Add nullable vector field ===" << std::endl;
    client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));

    auto schema = std::make_shared<milvus::CollectionSchema>();
    schema->AddField(milvus::FieldSchema(field_id, milvus::DataType::INT64, "", true, false));
    schema->AddField(milvus::FieldSchema(field_name, milvus::DataType::VARCHAR).WithMaxLength(100));
    schema->AddField(milvus::FieldSchema(field_vector_v1, milvus::DataType::FLOAT_VECTOR).WithDimension(DIMENSION));

    auto status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
    util::CheckStatus("create collection", status);

    status = client->CreateIndex(
        milvus::CreateIndexRequest()
            .WithCollectionName(collection_name)
            .AddIndex(milvus::IndexDesc(field_vector_v1, "", milvus::IndexType::FLAT, milvus::MetricType::L2)));
    util::CheckStatus("create initial index", status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection", status);

    milvus::EntityRows rows;
    for (int64_t i = 0; i < 10; ++i) {
        rows.emplace_back(milvus::EntityRow{{field_id, i},
                                            {field_name, "item_" + std::to_string(i)},
                                            {field_vector_v1, util::GenerateFloatVector(DIMENSION)}});
    }
    milvus::InsertResponse insert_response;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            insert_response);
    util::CheckStatus("insert initial rows", status);

    status = client->ReleaseCollection(milvus::ReleaseCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("release collection", status);

    auto nullable_vector = milvus::FieldSchema(field_vector_v2, milvus::DataType::FLOAT_VECTOR)
                               .WithDimension(DIMENSION)
                               .WithNullable(true);
    status = client->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithCollectionName(collection_name).WithField(std::move(nullable_vector)));
    util::CheckStatus("add nullable vector field", status);

    status = client->CreateIndex(
        milvus::CreateIndexRequest()
            .WithCollectionName(collection_name)
            .AddIndex(milvus::IndexDesc(field_vector_v2, "", milvus::IndexType::FLAT, milvus::MetricType::L2)));
    util::CheckStatus("create index for added field", status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("reload collection", status);

    milvus::QueryResponse query_response;
    status = client->Query(milvus::QueryRequest()
                               .WithCollectionName(collection_name)
                               .WithFilter(field_id + " >= 0")
                               .AddOutputField(field_id)
                               .AddOutputField(field_vector_v1)
                               .AddOutputField(field_vector_v2)
                               .WithLimit(10)
                               .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                           query_response);
    util::CheckStatus("query added nullable vector field", status);

    milvus::EntityRows output_rows;
    status = query_response.Results().OutputRows(output_rows);
    util::CheckStatus("get query output rows", status);
    std::cout << "Old rows after adding " << field_vector_v2 << ":" << std::endl;
    for (const auto& row : output_rows) {
        std::cout << "  id=" << row.at(field_id) << ", " << field_vector_v1 << "="
                  << (row.at(field_vector_v1).is_null() ? "null" : "has value") << ", " << field_vector_v2 << "="
                  << (row.at(field_vector_v2).is_null() ? "null" : "has value") << std::endl;
    }

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("drop collection", status);
}

}  // namespace

int
main() {
    std::cout << "Example start..." << std::endl;
    auto client = milvus::MilvusClientV2::Create();
    auto status = client->Connect({"http://localhost:19530", "root:Milvus"});
    util::CheckStatus("connect milvus server", status);

    InsertNullVectors(client);
    AddNullableVectorField(client);

    client->Disconnect();
    std::cout << "Done!" << std::endl;
    return 0;
}
