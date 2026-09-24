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
const std::string collection_name = "CPP_V2_ALIAS";
const std::string second_collection_name = collection_name + "_2";
const std::string alias_name = "cpp_v2_alias";
const std::string field_id = "id";
const std::string field_vector = "vector";
const uint32_t dimension = 4;

void
CreateCollection(milvus::MilvusClientV2Ptr& client, const std::string& name) {
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, true});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));

    // drop leftovers from a previous run; an alias blocks dropping its target collection,
    // so the alias (if any) must be dropped before either collection.
    client->DropAlias(milvus::DropAliasRequest().WithAlias(alias_name));
    client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(second_collection_name));
    client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(name));

    auto status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + name, status);
}

void
CreateAlias(milvus::MilvusClientV2Ptr& client) {
    auto status =
        client->CreateAlias(milvus::CreateAliasRequest().WithCollectionName(collection_name).WithAlias(alias_name));
    util::CheckStatus("create alias: " + alias_name, status);
}

void
ListAliases(milvus::MilvusClientV2Ptr& client) {
    milvus::ListAliasesResponse response;
    auto status = client->ListAliases(milvus::ListAliasesRequest().WithCollectionName(collection_name), response);
    util::CheckStatus("list aliases of: " + collection_name, status);
    std::cout << "Aliases of " << collection_name << ": ";
    util::PrintList(response.Aliases());
}

void
DescribeAlias(milvus::MilvusClientV2Ptr& client) {
    milvus::DescribeAliasResponse response;
    auto status = client->DescribeAlias(milvus::DescribeAliasRequest().WithAlias(alias_name), response);
    util::CheckStatus("describe alias: " + alias_name, status);
    std::cout << "Alias '" << response.Desc().Name() << "' -> collection '" << response.Desc().CollectionName()
              << "' in database '" << response.Desc().DatabaseName() << "'" << std::endl;
}

void
AlterAlias(milvus::MilvusClientV2Ptr& client) {
    // create a second collection and repoint the alias to it
    CreateCollection(client, second_collection_name);
    auto status = client->AlterAlias(
        milvus::AlterAliasRequest().WithCollectionName(second_collection_name).WithAlias(alias_name));
    util::CheckStatus("alter alias: " + alias_name, status);
}

void
DropAlias(milvus::MilvusClientV2Ptr& client) {
    auto status = client->DropAlias(milvus::DropAliasRequest().WithAlias(alias_name));
    util::CheckStatus("drop alias: " + alias_name, status);
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    CreateCollection(client, collection_name);
    CreateAlias(client);
    ListAliases(client);
    DescribeAlias(client);
    AlterAlias(client);
    DescribeAlias(client);
    DropAlias(client);

    // cleanup
    client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(second_collection_name));
    client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));

    client->Disconnect();
    return 0;
}
