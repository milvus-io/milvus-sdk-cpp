// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <memory>
#include <tuple>

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"

using ::testing::_;
using ::testing::AllOf;
using ::testing::Combine;
using ::testing::InSequence;
using ::testing::Property;
using ::testing::Values;

namespace {

enum class AliasMutation { CREATE, ALTER, DROP };

std::shared_ptr<milvus::MilvusClientV2>
CreateConnectedV2Client(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ::milvus::proto::milvus::ConnectRequest*,
                     ::milvus::proto::milvus::ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    auto status = client->Connect(milvus::ConnectParam{"127.0.0.1", port});
    EXPECT_TRUE(status.IsOk());
    return client;
}

void
SetCollectionID(milvus::proto::milvus::DescribeCollectionResponse* response, int64_t collection_id) {
    response->set_collectionid(collection_id);
}

void
ExpectV2AliasMutation(testing::StrictMock<::milvus::MilvusMockedService>& service, AliasMutation mutation,
                      const std::string& collection_name, const std::string& alias,
                      ::grpc::Status rpc_status = ::grpc::Status{}) {
    switch (mutation) {
        case AliasMutation::CREATE:
            EXPECT_CALL(service, CreateAlias(_,
                                             AllOf(Property(&milvus::proto::milvus::CreateAliasRequest::collection_name,
                                                            collection_name),
                                                   Property(&milvus::proto::milvus::CreateAliasRequest::alias, alias)),
                                             _))
                .WillOnce([rpc_status](::grpc::ServerContext*, const milvus::proto::milvus::CreateAliasRequest*,
                                       milvus::proto::common::Status*) { return rpc_status; });
            break;
        case AliasMutation::ALTER:
            EXPECT_CALL(
                service,
                AlterAlias(_,
                           AllOf(Property(&milvus::proto::milvus::AlterAliasRequest::collection_name, collection_name),
                                 Property(&milvus::proto::milvus::AlterAliasRequest::alias, alias)),
                           _))
                .WillOnce([rpc_status](::grpc::ServerContext*, const milvus::proto::milvus::AlterAliasRequest*,
                                       milvus::proto::common::Status*) { return rpc_status; });
            break;
        case AliasMutation::DROP:
            EXPECT_CALL(service, DropAlias(_, Property(&milvus::proto::milvus::DropAliasRequest::alias, alias), _))
                .WillOnce([rpc_status](::grpc::ServerContext*, const milvus::proto::milvus::DropAliasRequest*,
                                       milvus::proto::common::Status*) { return rpc_status; });
            break;
    }
}

milvus::Status
RunV2AliasMutation(const std::shared_ptr<milvus::MilvusClientV2>& client, AliasMutation mutation,
                   const std::string& collection_name, const std::string& alias) {
    switch (mutation) {
        case AliasMutation::CREATE:
            return client->CreateAlias(
                milvus::CreateAliasRequest().WithCollectionName(collection_name).WithAlias(alias));
        case AliasMutation::ALTER:
            return client->AlterAlias(milvus::AlterAliasRequest().WithCollectionName(collection_name).WithAlias(alias));
        case AliasMutation::DROP:
            return client->DropAlias(milvus::DropAliasRequest().WithAlias(alias));
    }
    return {milvus::StatusCode::UNKNOWN_ERROR, "Unknown alias mutation"};
}

class V2AliasCacheTest : public UnconnectMilvusMockedTest, public testing::WithParamInterface<AliasMutation> {};

TEST_P(V2AliasCacheTest, SuccessfulMutationInvalidatesCollectionDescCache) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    const std::string alias = "active_collection";
    const std::string target = "new_collection";

    InSequence sequence;
    EXPECT_CALL(
        service_,
        DescribeCollection(_, Property(&milvus::proto::milvus::DescribeCollectionRequest::collection_name, alias), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetCollectionID(response, 100);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_,
                ManualCompaction(_, Property(&milvus::proto::milvus::ManualCompactionRequest::collectionid, 100), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ManualCompactionRequest*,
                     milvus::proto::milvus::ManualCompactionResponse*) { return ::grpc::Status{}; });
    ExpectV2AliasMutation(service_, GetParam(), target, alias);
    EXPECT_CALL(
        service_,
        DescribeCollection(_, Property(&milvus::proto::milvus::DescribeCollectionRequest::collection_name, alias), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetCollectionID(response, 200);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_,
                ManualCompaction(_, Property(&milvus::proto::milvus::ManualCompactionRequest::collectionid, 200), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ManualCompactionRequest*,
                     milvus::proto::milvus::ManualCompactionResponse*) { return ::grpc::Status{}; });

    milvus::CompactResponse response;
    auto status = client->Compact(milvus::CompactRequest().WithCollectionName(alias), response);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    status = RunV2AliasMutation(client, GetParam(), target, alias);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    status = client->Compact(milvus::CompactRequest().WithCollectionName(alias), response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
}

INSTANTIATE_TEST_SUITE_P(AllAliasMutations, V2AliasCacheTest,
                         testing::Values(AliasMutation::CREATE, AliasMutation::ALTER, AliasMutation::DROP));

class V2AmbiguousAliasCacheTest : public UnconnectMilvusMockedTest,
                                  public testing::WithParamInterface<std::tuple<AliasMutation, ::grpc::StatusCode>> {};

TEST_P(V2AmbiguousAliasCacheTest, MutationInvalidatesCollectionDescCache) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    const std::string alias = "active_collection";
    const auto mutation = std::get<0>(GetParam());
    const auto rpc_code = std::get<1>(GetParam());

    InSequence sequence;
    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetCollectionID(response, 100);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, ManualCompaction(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ManualCompactionRequest*,
                     milvus::proto::milvus::ManualCompactionResponse*) { return ::grpc::Status{}; });
    ExpectV2AliasMutation(service_, mutation, "new_collection", alias,
                          ::grpc::Status{rpc_code, "ambiguous transport failure"});
    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetCollectionID(response, 200);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, ManualCompaction(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ManualCompactionRequest*,
                     milvus::proto::milvus::ManualCompactionResponse*) { return ::grpc::Status{}; });

    milvus::CompactResponse response;
    auto status = client->Compact(milvus::CompactRequest().WithCollectionName(alias), response);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    status = RunV2AliasMutation(client, mutation, "new_collection", alias);
    ASSERT_FALSE(status.IsOk());
    EXPECT_EQ(rpc_code == ::grpc::StatusCode::DEADLINE_EXCEEDED ? milvus::StatusCode::TIMEOUT
                                                                : milvus::StatusCode::RPC_FAILED,
              status.Code());
    status = client->Compact(milvus::CompactRequest().WithCollectionName(alias), response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
}

INSTANTIATE_TEST_SUITE_P(AllAliasMutationsAndTransportFailures, V2AmbiguousAliasCacheTest,
                         Combine(Values(AliasMutation::CREATE, AliasMutation::ALTER, AliasMutation::DROP),
                                 Values(::grpc::StatusCode::UNKNOWN, ::grpc::StatusCode::DEADLINE_EXCEEDED)));

TEST_F(UnconnectMilvusMockedTest, RejectedV2AliasMutationKeepsCollectionDescCache) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    const std::string alias = "active_collection";

    InSequence sequence;
    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetCollectionID(response, 100);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, ManualCompaction(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ManualCompactionRequest*,
                     milvus::proto::milvus::ManualCompactionResponse*) { return ::grpc::Status{}; });
    EXPECT_CALL(service_, AlterAlias(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::AlterAliasRequest*,
                     milvus::proto::common::Status* response) {
            response->set_code(1);
            response->set_reason("alter rejected");
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, ManualCompaction(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ManualCompactionRequest*,
                     milvus::proto::milvus::ManualCompactionResponse*) { return ::grpc::Status{}; });

    milvus::CompactResponse response;
    auto status = client->Compact(milvus::CompactRequest().WithCollectionName(alias), response);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    status = client->AlterAlias(milvus::AlterAliasRequest().WithCollectionName("new_collection").WithAlias(alias));
    ASSERT_FALSE(status.IsOk());
    EXPECT_EQ(milvus::StatusCode::SERVER_FAILED, status.Code());
    status = client->Compact(milvus::CompactRequest().WithCollectionName(alias), response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
}

TEST_F(UnconnectMilvusMockedTest, RateLimitRetryTimeoutKeepsCollectionDescCache) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::RetryParam retry_param;
    auto status = client->SetRetryParam(retry_param.WithMaxRetryTimes(2).WithInitialBackOffMs(1).WithMaxBackOffMs(1));
    ASSERT_TRUE(status.IsOk()) << status.Message();
    const std::string alias = "active_collection";

    InSequence sequence;
    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetCollectionID(response, 100);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, ManualCompaction(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ManualCompactionRequest*,
                     milvus::proto::milvus::ManualCompactionResponse*) { return ::grpc::Status{}; });
    EXPECT_CALL(service_, AlterAlias(_, _, _))
        .Times(2)
        .WillRepeatedly([](::grpc::ServerContext*, const milvus::proto::milvus::AlterAliasRequest*,
                           milvus::proto::common::Status* response) {
            response->set_code(8);
            response->set_reason("rate limited");
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, ManualCompaction(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ManualCompactionRequest*,
                     milvus::proto::milvus::ManualCompactionResponse*) { return ::grpc::Status{}; });

    milvus::CompactResponse response;
    status = client->Compact(milvus::CompactRequest().WithCollectionName(alias), response);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    status = client->AlterAlias(milvus::AlterAliasRequest().WithCollectionName("new_collection").WithAlias(alias));
    ASSERT_FALSE(status.IsOk());
    EXPECT_EQ(milvus::StatusCode::TIMEOUT, status.Code());
    EXPECT_EQ(0, status.RpcErrCode());
    EXPECT_EQ(8, status.ServerCode());
    status = client->Compact(milvus::CompactRequest().WithCollectionName(alias), response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
}

TEST_F(UnconnectMilvusMockedTest, RenameAndRecreateInvalidatesCollectionDescCache) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    const std::string old_name = "old_collection";
    const std::string new_name = "new_collection";

    InSequence sequence;
    EXPECT_CALL(service_,
                DescribeCollection(
                    _, Property(&milvus::proto::milvus::DescribeCollectionRequest::collection_name, old_name), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetCollectionID(response, 100);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_,
                ManualCompaction(_, Property(&milvus::proto::milvus::ManualCompactionRequest::collectionid, 100), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ManualCompactionRequest*,
                     milvus::proto::milvus::ManualCompactionResponse*) { return ::grpc::Status{}; });
    EXPECT_CALL(service_,
                RenameCollection(_,
                                 AllOf(Property(&milvus::proto::milvus::RenameCollectionRequest::oldname, old_name),
                                       Property(&milvus::proto::milvus::RenameCollectionRequest::newname, new_name)),
                                 _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::RenameCollectionRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });
    EXPECT_CALL(service_,
                DescribeCollection(
                    _, Property(&milvus::proto::milvus::DescribeCollectionRequest::collection_name, old_name), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse*) {
            return ::grpc::Status{::grpc::StatusCode::NOT_FOUND, "collection not found"};
        });
    EXPECT_CALL(
        service_,
        CreateCollection(_, Property(&milvus::proto::milvus::CreateCollectionRequest::collection_name, old_name), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::CreateCollectionRequest*,
                     milvus::proto::common::Status*) { return ::grpc::Status{}; });
    EXPECT_CALL(service_,
                DescribeCollection(
                    _, Property(&milvus::proto::milvus::DescribeCollectionRequest::collection_name, old_name), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetCollectionID(response, 200);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_,
                ManualCompaction(_, Property(&milvus::proto::milvus::ManualCompactionRequest::collectionid, 200), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ManualCompactionRequest*,
                     milvus::proto::milvus::ManualCompactionResponse*) { return ::grpc::Status{}; });

    milvus::CompactResponse compact_response;
    auto status = client->Compact(milvus::CompactRequest().WithCollectionName(old_name), compact_response);
    ASSERT_TRUE(status.IsOk()) << status.Message();

    status = client->RenameCollection(
        milvus::RenameCollectionRequest().WithCollectionName(old_name).WithNewCollectionName(new_name));
    ASSERT_TRUE(status.IsOk()) << status.Message();

    status = client->Compact(milvus::CompactRequest().WithCollectionName(old_name), compact_response);
    ASSERT_FALSE(status.IsOk());

    auto schema = std::make_shared<milvus::CollectionSchema>();
    schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "", true));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(old_name).WithCollectionSchema(std::move(schema)));
    ASSERT_TRUE(status.IsOk()) << status.Message();

    status = client->Compact(milvus::CompactRequest().WithCollectionName(old_name), compact_response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
}

}  // namespace
