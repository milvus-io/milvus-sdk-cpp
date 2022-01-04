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

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::AlterAliasRequest;
using ::milvus::proto::milvus::CreateAliasRequest;
using ::milvus::proto::milvus::DropAliasRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, CreateAliasFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection_name = "test";
    const std::string alias = "alias";

    EXPECT_CALL(service_, CreateAlias(_,
                                      AllOf(Property(&CreateAliasRequest::collection_name, collection_name),
                                            Property(&CreateAliasRequest::alias, alias)),
                                      _))
        .WillOnce([](::grpc::ServerContext*, const CreateAliasRequest* request, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    auto status = client_->CreateAlias(collection_name, alias);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, CreateAliasFooWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    const std::string collection_name = "test";
    const std::string alias = "alias";

    auto status = client_->CreateAlias(collection_name, alias);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, CreateAliasFooFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection_name = "test";
    const std::string alias = "alias";

    EXPECT_CALL(service_, CreateAlias(_,
                                      AllOf(Property(&CreateAliasRequest::collection_name, collection_name),
                                            Property(&CreateAliasRequest::alias, alias)),
                                      _))
        .WillOnce(
            [](::grpc::ServerContext*, const CreateAliasRequest* request, ::milvus::proto::common::Status* status) {
                return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
            });

    auto status = client_->CreateAlias(collection_name, alias);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(MilvusMockedTest, DropAliasFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string alias = "alias";

    EXPECT_CALL(service_, DropAlias(_, Property(&DropAliasRequest::alias, alias), _))
        .WillOnce([](::grpc::ServerContext*, const DropAliasRequest* request, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    auto status = client_->DropAlias(alias);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, DropAliasFooWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    const std::string alias = "alias";

    auto status = client_->DropAlias(alias);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, DropAliasFooFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string alias = "alias";

    EXPECT_CALL(service_, DropAlias(_, Property(&DropAliasRequest::alias, alias), _))
        .WillOnce([](::grpc::ServerContext*, const DropAliasRequest* request, ::milvus::proto::common::Status* status) {
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
        });

    auto status = client_->DropAlias(alias);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(MilvusMockedTest, AlterAliasFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection_name = "test";
    const std::string alias = "alias";

    EXPECT_CALL(service_, AlterAlias(_,
                                     AllOf(Property(&AlterAliasRequest::collection_name, collection_name),
                                           Property(&AlterAliasRequest::alias, alias)),
                                     _))
        .WillOnce([](::grpc::ServerContext*, const AlterAliasRequest* request, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    auto status = client_->AlterAlias(collection_name, alias);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, AlterAliasFooWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    const std::string collection_name = "test";
    const std::string alias = "alias";

    auto status = client_->AlterAlias(collection_name, alias);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, AlterAliasFooFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection_name = "test";
    const std::string alias = "alias";

    EXPECT_CALL(service_, AlterAlias(_,
                                     AllOf(Property(&AlterAliasRequest::collection_name, collection_name),
                                           Property(&AlterAliasRequest::alias, alias)),
                                     _))
        .WillOnce(
            [](::grpc::ServerContext*, const AlterAliasRequest* request, ::milvus::proto::common::Status* status) {
                return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
            });

    auto status = client_->AlterAlias(collection_name, alias);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}
