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

using ::testing::_;
using ::testing::AllOf;
using ::testing::Combine;
using ::testing::InSequence;
using ::testing::Property;
using ::testing::Values;

namespace {

enum class AliasMutation { CREATE, ALTER, DROP };

void
SetInsertSchema(milvus::proto::milvus::DescribeCollectionResponse* response) {
    auto* field = response->mutable_schema()->add_fields();
    field->set_name("id");
    field->set_data_type(milvus::proto::schema::DataType::Int64);
    field->set_is_primary_key(true);
}

void
ExpectInsert(testing::StrictMock<::milvus::MilvusMockedService>& service, const std::string& collection_name) {
    EXPECT_CALL(service,
                Insert(_, Property(&milvus::proto::milvus::InsertRequest::collection_name, collection_name), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::InsertRequest*,
                     milvus::proto::milvus::MutationResult* response) {
            response->mutable_ids()->mutable_int_id()->add_data(1);
            response->set_insert_cnt(1);
            return ::grpc::Status{};
        });
}

milvus::Status
RunInsert(const milvus::MilvusClientPtr& client, const std::string& collection_name) {
    std::vector<milvus::FieldDataPtr> fields{std::make_shared<milvus::Int64FieldData>("id", std::vector<int64_t>{1})};
    milvus::DmlResults results;
    return client->Insert(collection_name, "", fields, results);
}

void
ExpectAliasMutation(testing::StrictMock<::milvus::MilvusMockedService>& service, AliasMutation mutation,
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
RunAliasMutation(const milvus::MilvusClientPtr& client, AliasMutation mutation, const std::string& collection_name,
                 const std::string& alias) {
    switch (mutation) {
        case AliasMutation::CREATE:
            return client->CreateAlias(collection_name, alias);
        case AliasMutation::ALTER:
            return client->AlterAlias(collection_name, alias);
        case AliasMutation::DROP:
            return client->DropAlias(alias);
    }
    return {milvus::StatusCode::UNKNOWN_ERROR, "Unknown alias mutation"};
}

class V1AliasCacheTest : public MilvusMockedTest, public testing::WithParamInterface<AliasMutation> {};

TEST_P(V1AliasCacheTest, SuccessfulMutationInvalidatesCollectionDescCache) {
    auto status = client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()});
    ASSERT_TRUE(status.IsOk()) << status.Message();
    const std::string alias = "active_collection";
    const std::string target = "new_collection";

    InSequence sequence;
    EXPECT_CALL(
        service_,
        DescribeCollection(_, Property(&milvus::proto::milvus::DescribeCollectionRequest::collection_name, alias), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetInsertSchema(response);
            return ::grpc::Status{};
        });
    ExpectInsert(service_, alias);
    ExpectAliasMutation(service_, GetParam(), target, alias);
    EXPECT_CALL(
        service_,
        DescribeCollection(_, Property(&milvus::proto::milvus::DescribeCollectionRequest::collection_name, alias), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetInsertSchema(response);
            return ::grpc::Status{};
        });
    ExpectInsert(service_, alias);

    status = RunInsert(client_, alias);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    status = RunAliasMutation(client_, GetParam(), target, alias);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    status = RunInsert(client_, alias);
    EXPECT_TRUE(status.IsOk()) << status.Message();
}

INSTANTIATE_TEST_SUITE_P(AllAliasMutations, V1AliasCacheTest,
                         testing::Values(AliasMutation::CREATE, AliasMutation::ALTER, AliasMutation::DROP));

class V1AmbiguousAliasCacheTest : public MilvusMockedTest,
                                  public testing::WithParamInterface<std::tuple<AliasMutation, ::grpc::StatusCode>> {};

TEST_P(V1AmbiguousAliasCacheTest, MutationInvalidatesCollectionDescCache) {
    auto status = client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()});
    ASSERT_TRUE(status.IsOk()) << status.Message();
    const std::string alias = "active_collection";
    const auto mutation = std::get<0>(GetParam());
    const auto rpc_code = std::get<1>(GetParam());

    InSequence sequence;
    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetInsertSchema(response);
            return ::grpc::Status{};
        });
    ExpectInsert(service_, alias);
    ExpectAliasMutation(service_, mutation, "new_collection", alias,
                        ::grpc::Status{rpc_code, "ambiguous transport failure"});
    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetInsertSchema(response);
            return ::grpc::Status{};
        });
    ExpectInsert(service_, alias);

    status = RunInsert(client_, alias);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    status = RunAliasMutation(client_, mutation, "new_collection", alias);
    ASSERT_FALSE(status.IsOk());
    EXPECT_EQ(rpc_code == ::grpc::StatusCode::DEADLINE_EXCEEDED ? milvus::StatusCode::TIMEOUT
                                                                : milvus::StatusCode::RPC_FAILED,
              status.Code());
    status = RunInsert(client_, alias);
    EXPECT_TRUE(status.IsOk()) << status.Message();
}

INSTANTIATE_TEST_SUITE_P(AllAliasMutationsAndTransportFailures, V1AmbiguousAliasCacheTest,
                         Combine(Values(AliasMutation::CREATE, AliasMutation::ALTER, AliasMutation::DROP),
                                 Values(::grpc::StatusCode::UNKNOWN, ::grpc::StatusCode::DEADLINE_EXCEEDED)));

TEST_F(MilvusMockedTest, RejectedV1AliasMutationKeepsCollectionDescCache) {
    auto status = client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()});
    ASSERT_TRUE(status.IsOk()) << status.Message();
    const std::string alias = "active_collection";

    InSequence sequence;
    EXPECT_CALL(
        service_,
        DescribeCollection(_, Property(&milvus::proto::milvus::DescribeCollectionRequest::collection_name, alias), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetInsertSchema(response);
            return ::grpc::Status{};
        });
    ExpectInsert(service_, alias);
    EXPECT_CALL(service_, AlterAlias(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::AlterAliasRequest*,
                     milvus::proto::common::Status* response) {
            response->set_code(1);
            response->set_reason("alter rejected");
            return ::grpc::Status{};
        });
    ExpectInsert(service_, alias);

    status = RunInsert(client_, alias);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    status = client_->AlterAlias("new_collection", alias);
    ASSERT_FALSE(status.IsOk());
    EXPECT_EQ(milvus::StatusCode::SERVER_FAILED, status.Code());
    status = RunInsert(client_, alias);
    EXPECT_TRUE(status.IsOk()) << status.Message();
}

TEST_F(MilvusMockedTest, RateLimitRetryTimeoutKeepsCollectionDescCache) {
    auto status = client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()});
    ASSERT_TRUE(status.IsOk()) << status.Message();
    milvus::RetryParam retry_param;
    status = client_->SetRetryParam(retry_param.WithMaxRetryTimes(2).WithInitialBackOffMs(1).WithMaxBackOffMs(1));
    ASSERT_TRUE(status.IsOk()) << status.Message();
    const std::string alias = "active_collection";

    InSequence sequence;
    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetInsertSchema(response);
            return ::grpc::Status{};
        });
    ExpectInsert(service_, alias);
    EXPECT_CALL(service_, AlterAlias(_, _, _))
        .Times(2)
        .WillRepeatedly([](::grpc::ServerContext*, const milvus::proto::milvus::AlterAliasRequest*,
                           milvus::proto::common::Status* response) {
            response->set_code(8);
            response->set_reason("rate limited");
            return ::grpc::Status{};
        });
    ExpectInsert(service_, alias);

    status = RunInsert(client_, alias);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    status = client_->AlterAlias("new_collection", alias);
    ASSERT_FALSE(status.IsOk());
    EXPECT_EQ(milvus::StatusCode::TIMEOUT, status.Code());
    EXPECT_EQ(0, status.RpcErrCode());
    EXPECT_EQ(8, status.ServerCode());
    status = RunInsert(client_, alias);
    EXPECT_TRUE(status.IsOk()) << status.Message();
}

TEST_F(MilvusMockedTest, RenameAndRecreateInvalidatesCollectionDescCache) {
    auto status = client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()});
    ASSERT_TRUE(status.IsOk()) << status.Message();
    const std::string old_name = "old_collection";
    const std::string new_name = "new_collection";

    InSequence sequence;
    EXPECT_CALL(service_,
                DescribeCollection(
                    _, Property(&milvus::proto::milvus::DescribeCollectionRequest::collection_name, old_name), _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            SetInsertSchema(response);
            return ::grpc::Status{};
        });
    ExpectInsert(service_, old_name);
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
            SetInsertSchema(response);
            return ::grpc::Status{};
        });
    ExpectInsert(service_, old_name);

    status = RunInsert(client_, old_name);
    ASSERT_TRUE(status.IsOk()) << status.Message();

    status = client_->RenameCollection(old_name, new_name);
    ASSERT_TRUE(status.IsOk()) << status.Message();

    status = RunInsert(client_, old_name);
    ASSERT_FALSE(status.IsOk());

    milvus::CollectionSchema schema(old_name);
    schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "", true));
    status = client_->CreateCollection(schema);
    ASSERT_TRUE(status.IsOk()) << status.Message();

    status = RunInsert(client_, old_name);
    EXPECT_TRUE(status.IsOk()) << status.Message();
}

}  // namespace
