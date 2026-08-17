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
//
// Note: this example requires the Milvus server to have Storage V3 enabled
// (common.storage.useLoonFFI = true). Otherwise creating a collection with a
// TEXT field fails with:
//   "TEXT field requires StorageV3; enable common.storage.useLoonFFI: invalid parameter"

#include <iostream>
#include <string>
#include <thread>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {

const char* const collection_name = "CPP_V2_TEXT_FIELD";
const char* const field_id = "id";
const char* const field_vector = "vector";
const char* const field_body = "body";
const uint32_t dimension = 4;

void
insertText(milvus::MilvusClientV2Ptr& client, std::string body) {
    milvus::EntityRow row;
    row[field_vector] = util::GenerateFloatVector(dimension);
    row[field_body] = std::move(body);

    milvus::InsertResponse resp_insert;
    auto status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).AddRowData(std::move(row)),
                                 resp_insert);
    util::CheckStatus("insert row-based", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;
}

void
insertTextColumnBased(milvus::MilvusClientV2Ptr& client) {
    // TEXT field data is carried as std::string; TextFieldData is an alias of VarCharFieldData
    auto body_field = std::make_shared<milvus::TextFieldData>(
        field_body, std::vector<std::string>{"Milvus is an open-source vector database built for similarity search.",
                                             "TEXT fields store long source content without a fixed max_length."});
    auto vector_field = std::make_shared<milvus::FloatVecFieldData>(
        field_vector,
        std::vector<std::vector<float>>{util::GenerateFloatVector(dimension), util::GenerateFloatVector(dimension)});

    milvus::InsertResponse resp_insert;
    auto status = client->Insert(milvus::InsertRequest()
                                     .WithCollectionName(collection_name)
                                     .AddColumnData(body_field)
                                     .AddColumnData(vector_field),
                                 resp_insert);
    util::CheckStatus("insert column-based", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted by column-based." << std::endl;
}

void
queryText(milvus::MilvusClientV2Ptr& client) {
    // query all rows and decode the TEXT field values
    milvus::QueryResponse response;
    auto status = client->Query(milvus::QueryRequest()
                                    .WithCollectionName(collection_name)
                                    .AddOutputField(field_body)
                                    .WithFilter("id >= 0")
                                    .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                                response);
    util::CheckStatus("query", status);

    milvus::EntityRows output_rows;
    status = response.Results().OutputRows(output_rows);
    util::CheckStatus("get output rows", status);
    std::cout << "Query results (" << output_rows.size() << " rows):" << std::endl;
    for (const auto& row : output_rows) {
        std::cout << "\t" << row[field_body].get<std::string>() << std::endl;
    }
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->SetEnableDynamicField(false);
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, true});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema->AddField(milvus::FieldSchema(field_body, milvus::DataType::TEXT));

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + std::string(collection_name), status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + std::string(collection_name), status);

    // insert some rows
    insertText(client, "Milvus stores vector embeddings and scalar fields in collections.");
    insertText(client, "Long documents can be stored in a TEXT field without a length limit.");
    insertTextColumnBased(client);

    // query and read the TEXT values back
    queryText(client);

    client->Disconnect();
    return 0;
}
