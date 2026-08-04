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
#include "milvus/MilvusClientV2.h"
#include "utils/cache/CollectionTsCache.h"
#include "utils/cache/SchemaCache.h"

using ::testing::_;

namespace {

milvus::MilvusClientV2Ptr
CreateConnectedClient(testing::StrictMock<milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ConnectRequest*,
                     milvus::proto::milvus::ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    auto status = client->Connect(milvus::ConnectParam{"127.0.0.1", port});
    EXPECT_TRUE(status.IsOk());
    return client;
}

milvus::CollectionDescPtr
MakeCollectionDesc(int64_t id) {
    auto desc = std::make_shared<milvus::CollectionDesc>();
    desc->SetID(id);
    return desc;
}

std::string
Endpoint(uint16_t port) {
    return "127.0.0.1:" + std::to_string(port);
}

void
ExpectNotCached(const std::string& endpoint, const std::string& db_name, const std::string& collection_name) {
    milvus::CollectionDescPtr desc;
    EXPECT_FALSE(milvus::SchemaCache::GetInstance().Get(endpoint, db_name, collection_name, desc));
}

void
ExpectCached(const std::string& endpoint, const std::string& db_name, const std::string& collection_name) {
    milvus::CollectionDescPtr desc;
    EXPECT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint, db_name, collection_name, desc));
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, V2CreateCollectionInvalidatesSchemaAndPreservesTimestamp) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "collection", MakeCollectionDesc(1));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "collection", 100);
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, CreateCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::CreateCollectionRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    auto schema = std::make_shared<milvus::CollectionSchema>("collection");
    schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "", true));
    auto status = client->CreateCollection(milvus::CreateCollectionRequest()
                                               .WithDatabaseName("db")
                                               .WithCollectionName("collection")
                                               .WithCollectionSchema(schema));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "collection");
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "db", "collection"), 100);

    EXPECT_CALL(service_, Query(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::QueryRequest* request,
                     milvus::proto::milvus::QueryResults*) {
            EXPECT_EQ(request->guarantee_timestamp(), 100);
            return ::grpc::Status{};
        });
    milvus::QueryResponse query_response;
    status = client->Query(milvus::QueryRequest()
                               .WithDatabaseName("db")
                               .WithCollectionName("collection")
                               .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION),
                           query_response);
    EXPECT_TRUE(status.IsOk());
    milvus::CollectionTsCache::GetInstance().Invalidate(endpoint, "db", "collection");
}

TEST_F(UnconnectMilvusMockedTest, V2CreateAliasInvalidatesSchemaAndCopiesCollectionTimestamp) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "alias", MakeCollectionDesc(1));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "alias", 1);
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "collection", 100);
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, CreateAlias(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::CreateAliasRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    auto status = client->CreateAlias(
        milvus::CreateAliasRequest().WithDatabaseName("db").WithCollectionName("collection").WithAlias("alias"));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "alias");
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "db", "collection"), 100);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "db", "alias"), 100);
}

TEST_F(UnconnectMilvusMockedTest, V2SessionQueryThroughNewAliasUsesCollectionTimestamp) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "collection", 100);
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, CreateAlias(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::CreateAliasRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });
    ASSERT_TRUE(
        client
            ->CreateAlias(
                milvus::CreateAliasRequest().WithDatabaseName("db").WithCollectionName("collection").WithAlias("alias"))
            .IsOk());

    EXPECT_CALL(service_, Query(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::QueryRequest* request,
                     milvus::proto::milvus::QueryResults*) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collection_name(), "alias");
            EXPECT_EQ(request->guarantee_timestamp(), 100);
            return ::grpc::Status{};
        });

    milvus::QueryResponse response;
    auto status =
        client->Query(milvus::QueryRequest().WithDatabaseName("db").WithCollectionName("alias").WithConsistencyLevel(
                          milvus::ConsistencyLevel::SESSION),
                      response);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, V2AlterAliasInvalidatesSchemaAndCopiesCollectionTimestamp) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "alias", MakeCollectionDesc(1));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "alias", 1);
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "collection", 100);
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, AlterAlias(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::AlterAliasRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    auto status = client->AlterAlias(
        milvus::AlterAliasRequest().WithDatabaseName("db").WithCollectionName("collection").WithAlias("alias"));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "alias");
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "db", "collection"), 100);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "db", "alias"), 100);
}

TEST_F(UnconnectMilvusMockedTest, V2DropAliasInvalidatesAliasCaches) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "alias", MakeCollectionDesc(1));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "alias", 1);
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, DropAlias(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DropAliasRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    auto status = client->DropAlias(milvus::DropAliasRequest().WithDatabaseName("db").WithAlias("alias"));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "alias");
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "db", "alias"), 0);
}

TEST_F(UnconnectMilvusMockedTest, V2RenameCollectionInvalidatesSchemaAndMovesTimestamp) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "old", MakeCollectionDesc(1));
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "new", MakeCollectionDesc(2));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "old", 100);
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "new", 200);
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, RenameCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::RenameCollectionRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    auto status = client->RenameCollection(
        milvus::RenameCollectionRequest().WithDatabaseName("db").WithCollectionName("old").WithNewCollectionName(
            "new"));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "old");
    ExpectNotCached(endpoint, "db", "new");
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "db", "old"), 0);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "db", "new"), 200);
    milvus::CollectionTsCache::GetInstance().Invalidate(endpoint, "db", "new");
}

TEST_F(UnconnectMilvusMockedTest, V2RenameCollectionAcrossDatabasesInvalidatesSchemaAndMovesTimestamp) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "source-db", "old", MakeCollectionDesc(1));
    milvus::SchemaCache::GetInstance().Set(endpoint, "target-db", "new", MakeCollectionDesc(2));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "source-db", "old", 100);
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "target-db", "new", 200);
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, RenameCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::RenameCollectionRequest* request,
                     milvus::proto::common::Status*) {
            EXPECT_EQ(request->db_name(), "source-db");
            EXPECT_EQ(request->oldname(), "old");
            EXPECT_EQ(request->newname(), "new");
            EXPECT_EQ(request->newdbname(), "target-db");
            return ::grpc::Status{};
        });

    auto status = client->RenameCollection(milvus::RenameCollectionRequest()
                                               .WithDatabaseName("source-db")
                                               .WithCollectionName("old")
                                               .WithNewCollectionName("new")
                                               .WithTargetDatabaseName("target-db"));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "source-db", "old");
    ExpectNotCached(endpoint, "target-db", "new");
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "source-db", "old"), 0);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "target-db", "new"), 200);
    milvus::CollectionTsCache::GetInstance().Invalidate(endpoint, "target-db", "new");
}

TEST_F(UnconnectMilvusMockedTest, V2DropDatabaseInvalidatesDatabaseCaches) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "first", MakeCollectionDesc(1));
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "second", MakeCollectionDesc(2));
    milvus::SchemaCache::GetInstance().Set(endpoint, "other", "third", MakeCollectionDesc(3));
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "first", 1);
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "db", "second", 2);
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "other", "third", 3);
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, DropDatabase(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DropDatabaseRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    auto status = client->DropDatabase(milvus::DropDatabaseRequest().WithDatabaseName("db"));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "first");
    ExpectNotCached(endpoint, "db", "second");

    milvus::CollectionDescPtr desc;
    EXPECT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint, "other", "third", desc));
    milvus::SchemaCache::GetInstance().InvalidateDb(endpoint, "other");

    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "db", "first"), 0);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "db", "second"), 0);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "other", "third"), 3);
    milvus::CollectionTsCache::GetInstance().InvalidateDb(endpoint, "other");
}

TEST_F(UnconnectMilvusMockedTest, V2CollectionPropertiesInvalidateCacheForAllowInsertAutoId) {
    const auto endpoint = Endpoint(server_.ListenPort());
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, AlterCollection(_, _, _))
        .Times(4)
        .WillRepeatedly([](::grpc::ServerContext*, const milvus::proto::milvus::AlterCollectionRequest*,
                           milvus::proto::common::Status*) { return ::grpc::Status{}; });
    EXPECT_CALL(service_, AlterCollectionField(_, _, _))
        .Times(2)
        .WillRepeatedly([](::grpc::ServerContext*, const milvus::proto::milvus::AlterCollectionFieldRequest*,
                           milvus::proto::common::Status*) { return ::grpc::Status{}; });

    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "collection", MakeCollectionDesc(1));
    auto status = client->AlterCollectionProperties(milvus::AlterCollectionPropertiesRequest()
                                                        .WithDatabaseName("db")
                                                        .WithCollectionName("collection")
                                                        .AddProperty("key", "value"));
    EXPECT_TRUE(status.IsOk());
    ExpectCached(endpoint, "db", "collection");

    status = client->DropCollectionProperties(milvus::DropCollectionPropertiesRequest()
                                                  .WithDatabaseName("db")
                                                  .WithCollectionName("collection")
                                                  .AddPropertyKey("key"));
    EXPECT_TRUE(status.IsOk());
    ExpectCached(endpoint, "db", "collection");

    status = client->AlterCollectionProperties(milvus::AlterCollectionPropertiesRequest()
                                                    .WithDatabaseName("db")
                                                    .WithCollectionName("collection")
                                                    .AddProperty("allow_insert_auto_id", "true"));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "collection");

    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "collection", MakeCollectionDesc(1));
    status = client->DropCollectionProperties(milvus::DropCollectionPropertiesRequest()
                                                  .WithDatabaseName("db")
                                                  .WithCollectionName("collection")
                                                  .AddPropertyKey("allow_insert_auto_id"));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "collection");

    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "collection", MakeCollectionDesc(1));
    status = client->AlterCollectionFieldProperties(milvus::AlterCollectionFieldPropertiesRequest()
                                                        .WithDatabaseName("db")
                                                        .WithCollectionName("collection")
                                                        .WithFieldName("field")
                                                        .AddProperty("key", "value"));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "collection");

    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "collection", MakeCollectionDesc(1));
    status = client->DropCollectionFieldProperties(milvus::DropCollectionFieldPropertiesRequest()
                                                       .WithDatabaseName("db")
                                                       .WithCollectionName("collection")
                                                       .WithFieldName("field")
                                                       .AddPropertyKey("key"));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "collection");
}

TEST_F(UnconnectMilvusMockedTest, V2StructAndFunctionMutationsInvalidateCollectionSchema) {
    const auto endpoint = Endpoint(server_.ListenPort());
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, AddCollectionStructField(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::AddCollectionStructFieldRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });
    EXPECT_CALL(service_, AddCollectionFunction(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::AddCollectionFunctionRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });
    EXPECT_CALL(service_, AlterCollectionFunction(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::AlterCollectionFunctionRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });
    EXPECT_CALL(service_, DropCollectionFunction(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DropCollectionFunctionRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });

    milvus::StructFieldSchema struct_field;
    struct_field.WithName("structs").WithMaxCapacity(8).WithNullable(true).AddField(
        milvus::FieldSchema("value", milvus::DataType::INT64));
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "collection", MakeCollectionDesc(1));
    auto status = client->AddCollectionStructField(milvus::AddCollectionStructFieldRequest()
                                                       .WithDatabaseName("db")
                                                       .WithCollectionName("collection")
                                                       .WithStructField(std::move(struct_field)));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "collection");

    auto function = std::make_shared<milvus::Function>("function", milvus::FunctionType::BM25);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("sparse");

    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "collection", MakeCollectionDesc(1));
    status = client->AddCollectionFunction(milvus::AddCollectionFunctionRequest()
                                               .WithDatabaseName("db")
                                               .WithCollectionName("collection")
                                               .WithFunction(function));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "collection");

    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "collection", MakeCollectionDesc(1));
    status = client->AlterCollectionFunction(milvus::AlterCollectionFunctionRequest()
                                                 .WithDatabaseName("db")
                                                 .WithCollectionName("collection")
                                                 .WithFunction(function));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "collection");

    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "collection", MakeCollectionDesc(1));
    status = client->DropCollectionFunction(milvus::DropCollectionFunctionRequest()
                                                .WithDatabaseName("db")
                                                .WithCollectionName("collection")
                                                .WithFunctionName("function"));
    EXPECT_TRUE(status.IsOk());
    ExpectNotCached(endpoint, "db", "collection");
}

TEST_F(UnconnectMilvusMockedTest, V2CompactBypassesSchemaCache) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "collection", MakeCollectionDesc(100));
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest* request,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collection_name(), "collection");
            response->set_collectionid(200);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, ManualCompaction(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ManualCompactionRequest* request,
                     milvus::proto::milvus::ManualCompactionResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collectionid(), 200);
            response->set_compactionid(10);
            return ::grpc::Status{};
        });

    milvus::CompactResponse response;
    auto status =
        client->Compact(milvus::CompactRequest().WithDatabaseName("db").WithCollectionName("collection"), response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.CompactionID(), 10);

    milvus::CollectionDescPtr cached;
    ASSERT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint, "db", "collection", cached));
    ASSERT_NE(cached, nullptr);
    EXPECT_EQ(cached->ID(), 100);
    milvus::SchemaCache::GetInstance().Invalidate(endpoint, "db", "collection");
}

TEST_F(UnconnectMilvusMockedTest, V2OptimizeDirectlyDescribesForInitializationAndCompaction) {
    const auto endpoint = Endpoint(server_.ListenPort());
    milvus::SchemaCache::GetInstance().Set(endpoint, "db", "collection", MakeCollectionDesc(100));
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest* request,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            response->set_collectionid(200);
            return ::grpc::Status{};
        })
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest* request,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            response->set_collectionid(300);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, ManualCompaction(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ManualCompactionRequest* request,
                     milvus::proto::milvus::ManualCompactionResponse* response) {
            EXPECT_EQ(request->collectionid(), 300);
            response->set_compactionid(10);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, GetCompactionState(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::GetCompactionStateRequest*,
                     milvus::proto::milvus::GetCompactionStateResponse* response) {
            response->set_state(milvus::proto::common::CompactionState::Completed);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, GetLoadState(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::GetLoadStateRequest* request,
                     milvus::proto::milvus::GetLoadStateResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            response->set_state(milvus::proto::common::LoadState::LoadStateNotLoad);
            return ::grpc::Status{};
        });

    milvus::OptimizeTaskPtr task;
    auto status =
        client->Optimize(milvus::OptimizeRequest().WithDatabaseName("db").WithCollectionName("collection"), task);
    EXPECT_TRUE(status.IsOk());
    ASSERT_NE(task, nullptr);

    milvus::CollectionDescPtr cached;
    ASSERT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint, "db", "collection", cached));
    ASSERT_NE(cached, nullptr);
    EXPECT_EQ(cached->ID(), 100);
    milvus::SchemaCache::GetInstance().Invalidate(endpoint, "db", "collection");
}
