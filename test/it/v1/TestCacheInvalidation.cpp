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

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "../mocks/MilvusMockedTest.h"
#include "utils/cache/CollectionTsCache.h"
#include "utils/cache/SchemaCache.h"

using ::testing::_;

namespace {

std::string
Endpoint(uint16_t port) {
    return "127.0.0.1:" + std::to_string(port);
}

milvus::CollectionDescPtr
MakeCollectionDesc(int64_t id) {
    auto desc = std::make_shared<milvus::CollectionDesc>();
    desc->SetID(id);
    return desc;
}

void
ConnectClient(const milvus::MilvusClientPtr& client, uint16_t port) {
    auto status = client->Connect(milvus::ConnectParam{"127.0.0.1", port});
    EXPECT_TRUE(status.IsOk());
}

void
ExpectSchemaNotCached(const std::string& endpoint, const std::string& db_name, const std::string& collection_name) {
    milvus::CollectionDescPtr desc;
    EXPECT_FALSE(milvus::SchemaCache::GetInstance().Get(endpoint, db_name, collection_name, desc));
}

void
ExpectSchemaCached(const std::string& endpoint, const std::string& db_name, const std::string& collection_name) {
    milvus::CollectionDescPtr desc;
    EXPECT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint, db_name, collection_name, desc));
}

}  // namespace

TEST_F(MilvusMockedTest, V1CreateCollectionInvalidatesSchemaAndPreservesTimestamp) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "default", "collection", MakeCollectionDesc(1));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "default", "collection", 100);
    ConnectClient(client_, server_.ListenPort());

    EXPECT_CALL(service_, CreateCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::CreateCollectionRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    milvus::CollectionSchema schema("collection");
    schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "", true));
    auto status = client_->CreateCollection(schema);
    EXPECT_TRUE(status.IsOk());
    ExpectSchemaNotCached(endpoint, "default", "collection");
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "default", "collection"), 100);
    milvus::CollectionTsCache::GetInstance().Invalidate(endpoint, "default", "collection");
}

TEST_F(MilvusMockedTest, V1CreateAliasInvalidatesSchemaAndCopiesCollectionTimestamp) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "default", "alias", MakeCollectionDesc(1));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "default", "alias", 1);
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "default", "collection", 100);
    ConnectClient(client_, server_.ListenPort());

    EXPECT_CALL(service_, CreateAlias(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::CreateAliasRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    auto status = client_->CreateAlias("collection", "alias");
    EXPECT_TRUE(status.IsOk());
    ExpectSchemaNotCached(endpoint, "default", "alias");
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "default", "collection"), 100);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "default", "alias"), 100);
}

TEST_F(MilvusMockedTest, V1AlterAliasInvalidatesSchemaAndCopiesCollectionTimestamp) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "default", "alias", MakeCollectionDesc(1));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "default", "alias", 1);
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "default", "collection", 100);
    ConnectClient(client_, server_.ListenPort());

    EXPECT_CALL(service_, AlterAlias(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::AlterAliasRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    auto status = client_->AlterAlias("collection", "alias");
    EXPECT_TRUE(status.IsOk());
    ExpectSchemaNotCached(endpoint, "default", "alias");
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "default", "collection"), 100);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "default", "alias"), 100);
}

TEST_F(MilvusMockedTest, V1DropAliasInvalidatesAliasCaches) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "default", "alias", MakeCollectionDesc(1));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "default", "alias", 1);
    ConnectClient(client_, server_.ListenPort());

    EXPECT_CALL(service_, DropAlias(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DropAliasRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    auto status = client_->DropAlias("alias");
    EXPECT_TRUE(status.IsOk());
    ExpectSchemaNotCached(endpoint, "default", "alias");
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "default", "alias"), 0);
}

TEST_F(MilvusMockedTest, V1RenameCollectionInvalidatesSchemaAndMovesTimestamp) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "default", "old", MakeCollectionDesc(1));
    milvus::SchemaCache::GetInstance().Set(endpoint, "default", "new", MakeCollectionDesc(2));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "default", "old", 100);
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "default", "new", 200);
    ConnectClient(client_, server_.ListenPort());

    EXPECT_CALL(service_, RenameCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::RenameCollectionRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    auto status = client_->RenameCollection("old", "new");
    EXPECT_TRUE(status.IsOk());
    ExpectSchemaNotCached(endpoint, "default", "old");
    ExpectSchemaNotCached(endpoint, "default", "new");
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "default", "old"), 0);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "default", "new"), 200);
    milvus::CollectionTsCache::GetInstance().Invalidate(endpoint, "default", "new");
}

TEST_F(MilvusMockedTest, V1DropDatabaseInvalidatesDatabaseCaches) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "first", MakeCollectionDesc(1));
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "second", MakeCollectionDesc(2));
    milvus::SchemaCache::GetInstance().Set(endpoint, "other", "third", MakeCollectionDesc(3));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "first", 1);
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "second", 2);
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "other", "third", 3);
    ConnectClient(client_, server_.ListenPort());

    EXPECT_CALL(service_, DropDatabase(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DropDatabaseRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    auto status = client_->DropDatabase("db");
    EXPECT_TRUE(status.IsOk());
    ExpectSchemaNotCached(endpoint, "db", "first");
    ExpectSchemaNotCached(endpoint, "db", "second");

    milvus::CollectionDescPtr desc;
    EXPECT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint, "other", "third", desc));
    milvus::SchemaCache::GetInstance().InvalidateDb(endpoint, "other");

    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "db", "first"), 0);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "db", "second"), 0);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "other", "third"), 3);
    milvus::CollectionTsCache::GetInstance().InvalidateDb(endpoint, "other");
}

TEST_F(MilvusMockedTest, V1CollectionPropertiesInvalidateCacheForAllowInsertAutoId) {
    const auto endpoint = Endpoint(server_.ListenPort());
    ConnectClient(client_, server_.ListenPort());

    EXPECT_CALL(service_, AlterCollection(_, _, _))
        .Times(4)
        .WillRepeatedly([](::grpc::ServerContext*, const milvus::proto::milvus::AlterCollectionRequest*,
                           milvus::proto::common::Status*) { return ::grpc::Status{}; });
    EXPECT_CALL(service_, AlterCollectionField(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::AlterCollectionFieldRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    milvus::SchemaCache::GetInstance().Set(endpoint, "default", "collection", MakeCollectionDesc(1));
    auto status = client_->AlterCollectionProperties("collection", {{"key", "value"}});
    EXPECT_TRUE(status.IsOk());
    ExpectSchemaCached(endpoint, "default", "collection");

    status = client_->DropCollectionProperties("collection", {"key"});
    EXPECT_TRUE(status.IsOk());
    ExpectSchemaCached(endpoint, "default", "collection");

    status = client_->AlterCollectionProperties("collection", {{"allow_insert_auto_id", "true"}});
    EXPECT_TRUE(status.IsOk());
    ExpectSchemaNotCached(endpoint, "default", "collection");

    milvus::SchemaCache::GetInstance().Set(endpoint, "default", "collection", MakeCollectionDesc(1));
    status = client_->DropCollectionProperties("collection", {"allow_insert_auto_id"});
    EXPECT_TRUE(status.IsOk());
    ExpectSchemaNotCached(endpoint, "default", "collection");

    milvus::SchemaCache::GetInstance().Set(endpoint, "default", "collection", MakeCollectionDesc(1));
    status = client_->AlterCollectionField("collection", "field", {{"key", "value"}});
    EXPECT_TRUE(status.IsOk());
    ExpectSchemaNotCached(endpoint, "default", "collection");
}
