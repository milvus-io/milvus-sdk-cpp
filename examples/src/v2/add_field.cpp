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
#include <memory>
#include <string>
#include <vector>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {
const std::string collection_name = "CPP_V2_ADD_FIELD";
const std::string field_id = "id";
const std::string field_vector = "vector";
const std::string field_text = "text";
const std::string field_sparse = "sparse";
const std::string function_name = "bm25";
const uint32_t dimension = 8;

void
CreateCollection(milvus::MilvusClientV2Ptr& client) {
    auto collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, false});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));

    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::COSINE);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + collection_name, status);

    std::cout << "Collection created" << std::endl;
}

void
InsertRow(milvus::MilvusClientV2Ptr& client, int64_t id, const std::string* text) {
    milvus::EntityRow row;
    row[field_id] = id;
    row[field_vector] = util::GenerateFloatVector(dimension);
    if (text != nullptr) {
        row[field_text] = *text;
    }

    milvus::EntityRows rows;
    rows.emplace_back(std::move(row));

    milvus::InsertResponse response;
    auto status = client->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), response);
    util::CheckStatus("insert", status);
}

void
QueryById(milvus::MilvusClientV2Ptr& client, int64_t id) {
    auto request = milvus::QueryRequest()
                       .WithCollectionName(collection_name)
                       .AddOutputField("*")
                       .WithFilter(field_id + " == " + std::to_string(id))
                       .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    std::cout << "\nQuery with id: " << id << std::endl;
    milvus::QueryResponse response;
    auto status = client->Query(request, response);
    util::CheckStatus("query", status);

    milvus::EntityRows output_rows;
    status = response.Results().OutputRows(output_rows);
    util::CheckStatus("get output rows", status);
    for (const auto& row : output_rows) {
        std::cout << row << std::endl;
    }
    std::cout << "=============================================================" << std::endl;
}

void
DescribeCollection(milvus::MilvusClientV2Ptr& client) {
    milvus::DescribeCollectionResponse response;
    auto status =
        client->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name), response);
    util::CheckStatus("describe collection", status);

    std::cout << "\nCollection fields:" << std::endl;
    for (const auto& field : response.Desc().Schema().Fields()) {
        std::cout << "  " << field.Name() << std::endl;
    }
    for (const auto& function : response.Desc().Schema().Functions()) {
        std::cout << "  function: " << function->Name() << std::endl;
    }
    std::cout << "=============================================================" << std::endl;
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    CreateCollection(client);

    InsertRow(client, 100, nullptr);

    {
        milvus::FieldSchema text_field =
            milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(100).WithNullable(true);
        status = client->AddCollectionField(
            milvus::AddCollectionFieldRequest().WithCollectionName(collection_name).WithField(std::move(text_field)));
        util::CheckStatus("add field 'text'", status);

        DescribeCollection(client);
        QueryById(client, 100);
    }

    {
        const std::string text_value = "this is a new row";
        InsertRow(client, 500, &text_value);
        QueryById(client, 500);
    }

    {
        status = client->DropCollectionField(
            milvus::DropCollectionFieldRequest().WithCollectionName(collection_name).WithFieldName(field_text));
        util::CheckStatus("drop field 'text'", status);
        std::cout << "Field 'text' dropped" << std::endl;
        DescribeCollection(client);
    }

    {
        milvus::FieldSchema text_field = milvus::FieldSchema(field_text, milvus::DataType::VARCHAR)
                                             .WithMaxLength(100)
                                             .EnableAnalyzer(true)
                                             .EnableMatch(true)
                                             .WithNullable(true);
        status = client->AddCollectionField(
            milvus::AddCollectionFieldRequest().WithCollectionName(collection_name).WithField(std::move(text_field)));
        util::CheckStatus("add field 'text' for function demo", status);

        milvus::FieldSchema sparse_field(field_sparse, milvus::DataType::SPARSE_FLOAT_VECTOR);
        milvus::FunctionPtr function = std::make_shared<milvus::Function>(function_name, milvus::FunctionType::BM25);
        function->AddInputFieldName(field_text);
        function->AddOutputFieldName(field_sparse);

        status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                              .WithCollectionName(collection_name)
                                              .WithField(std::move(sparse_field))
                                              .WithFunction(function));
        util::CheckStatus("add function-backed field 'sparse'", status);
        std::cout << "Function-backed field 'sparse' added" << std::endl;
        DescribeCollection(client);
    }

    {
        status = client->DropFunctionField(
            milvus::DropFunctionFieldRequest().WithCollectionName(collection_name).WithFunctionName(function_name));
        util::CheckStatus("drop function-backed field 'sparse'", status);
        std::cout << "Function-backed field 'sparse' dropped" << std::endl;
        DescribeCollection(client);
    }

    client->Disconnect();
    return 0;
}
