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
const std::string collection_name = "CPP_V2_INDEX";
const std::string field_id = "id";
const std::string field_vector = "vector";
const std::string field_name = "name";
const uint32_t dimension = 128;
const std::string vector_index_name = "idx_vector";
const std::string scalar_index_name = "idx_scalar";

void
CreateCollection(milvus::MilvusClientV2Ptr& client) {
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, true});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    milvus::FieldSchema name_schema{field_name, milvus::DataType::VARCHAR};
    name_schema.SetMaxLength(256);
    collection_schema->AddField(name_schema);

    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);
}

void
CreateIndexes(milvus::MilvusClientV2Ptr& client) {
    milvus::IndexParam index_vector(field_vector, vector_index_name, milvus::IndexType::IVF_FLAT,
                                    milvus::MetricType::L2);
    index_vector.AddExtraParam(milvus::NLIST, "128");

    milvus::IndexParam index_scalar(field_name, scalar_index_name, milvus::IndexType::INVERTED);

    auto status = client->CreateIndex(milvus::CreateIndexRequest()
                                          .WithCollectionName(collection_name)
                                          .WithSync(true)
                                          .AddIndexParam(std::move(index_vector))
                                          .AddIndexParam(std::move(index_scalar)));
    util::CheckStatus("create indexes on collection: " + collection_name, status);
}

void
ListIndexes(milvus::MilvusClientV2Ptr& client) {
    milvus::ListIndexesResponse response;
    auto status = client->ListIndexes(milvus::ListIndexesRequest().WithCollectionName(collection_name), response);
    util::CheckStatus("list indexes of collection: " + collection_name, status);
    std::cout << "Index names: ";
    util::PrintList(response.IndexNames());
}

void
PrintIndexDesc(const milvus::DescribeIndexResponse& response) {
    for (const auto& desc : response.Descs()) {
        std::cout << "  Index '" << desc.IndexName() << "' on field '" << desc.FieldName()
                  << "': type=" << std::to_string(desc.IndexType()) << ", metric=" << std::to_string(desc.MetricType())
                  << ", state=" << std::to_string(desc.StateCode()) << ", rows=" << desc.IndexedRows() << "/"
                  << desc.TotalRows() << std::endl;
    }
}

void
DescribeIndexes(milvus::MilvusClientV2Ptr& client) {
    milvus::DescribeIndexResponse response;
    auto status = client->DescribeIndex(
        milvus::DescribeIndexRequest().WithCollectionName(collection_name).WithIndexName(vector_index_name), response);
    util::CheckStatus("describe index by index name: " + vector_index_name, status);
    PrintIndexDesc(response);

    status = client->DescribeIndex(
        milvus::DescribeIndexRequest().WithCollectionName(collection_name).WithFieldName(field_name), response);
    util::CheckStatus("describe index by field name: " + field_name, status);
    PrintIndexDesc(response);
}

void
AlterIndexProperties(milvus::MilvusClientV2Ptr& client) {
    auto status = client->AlterIndexProperties(milvus::AlterIndexPropertiesRequest()
                                                   .WithCollectionName(collection_name)
                                                   .WithIndexName(vector_index_name)
                                                   .AddProperty(milvus::MMAP_ENABLED, "true"));
    util::CheckStatus("alter index properties of: " + vector_index_name, status);
}

void
DropIndexProperties(milvus::MilvusClientV2Ptr& client) {
    auto status = client->DropIndexProperties(milvus::DropIndexPropertiesRequest()
                                                  .WithCollectionName(collection_name)
                                                  .WithIndexName(vector_index_name)
                                                  .AddPropertyKey(milvus::MMAP_ENABLED));
    util::CheckStatus("drop index properties of: " + vector_index_name, status);
}

void
DropIndexes(milvus::MilvusClientV2Ptr& client) {
    auto status = client->DropIndex(
        milvus::DropIndexRequest().WithCollectionName(collection_name).WithIndexName(vector_index_name));
    util::CheckStatus("drop index: " + vector_index_name, status);

    status = client->DropIndex(
        milvus::DropIndexRequest().WithCollectionName(collection_name).WithIndexName(scalar_index_name));
    util::CheckStatus("drop index: " + scalar_index_name, status);
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
    CreateIndexes(client);
    ListIndexes(client);
    DescribeIndexes(client);
    AlterIndexProperties(client);
    DropIndexProperties(client);
    DropIndexes(client);

    client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    client->Disconnect();
    return 0;
}
