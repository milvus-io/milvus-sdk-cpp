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

using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::OperatePrivilegeRequest;
using ::milvus::proto::milvus::UpdateCredentialRequest;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Property;

namespace {

std::shared_ptr<milvus::MilvusClientV2>
CreateConnectedV2Client(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{"127.0.0.1", port};
    auto status = client->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());
    return client;
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, GrantPrivilege) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, OperatePrivilege(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const OperatePrivilegeRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->entity().role().name(), "reader_role");
            EXPECT_EQ(request->entity().object().name(), "Collection");
            EXPECT_EQ(request->entity().object_name(), "my_coll");
            EXPECT_EQ(request->entity().grantor().privilege().name(), "Insert");
            EXPECT_EQ(request->entity().db_name(), "my_db");
            EXPECT_EQ(request->type(), ::milvus::proto::milvus::OperatePrivilegeType::Grant);
            return ::grpc::Status{};
        });

    auto status = client->GrantPrivilege(milvus::GrantPrivilegeRequest()
                                             .WithRoleName("reader_role")
                                             .WithObjectType("Collection")
                                             .WithObjectName("my_coll")
                                             .WithPrivilege("Insert")
                                             .WithDatabaseName("my_db"));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, RevokePrivilege) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, OperatePrivilege(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const OperatePrivilegeRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->entity().role().name(), "reader_role");
            EXPECT_EQ(request->entity().object().name(), "Global");
            EXPECT_EQ(request->entity().object_name(), "");
            EXPECT_EQ(request->entity().grantor().privilege().name(), "CreateCollection");
            EXPECT_TRUE(request->entity().db_name().empty());
            EXPECT_EQ(request->type(), ::milvus::proto::milvus::OperatePrivilegeType::Revoke);
            return ::grpc::Status{};
        });

    auto status = client->RevokePrivilege(milvus::RevokePrivilegeRequest()
                                              .WithRoleName("reader_role")
                                              .WithObjectType("Global")
                                              .WithPrivilege("CreateCollection"));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, UpdatePasswordWithDescription) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, UpdateCredential(_,
                                           AllOf(Property(&UpdateCredentialRequest::username, "username"),
                                                 Property(&UpdateCredentialRequest::oldpassword, "b2xk"),
                                                 Property(&UpdateCredentialRequest::newpassword, "bmV3"),
                                                 Property(&UpdateCredentialRequest::description, "new description")),
                                           _))
        .WillOnce([](::grpc::ServerContext*, const UpdateCredentialRequest*, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    auto status = client->UpdatePassword(milvus::UpdatePasswordRequest()
                                             .WithUserName("username")
                                             .WithOldPassword("old")
                                             .WithNewPassword("new")
                                             .WithDescription("new description"));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, UpdatePasswordResetConnection) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    // The initial Connect is matched by CreateConnectedV2Client; the credential-reset
    // reconnect issues a second Connect RPC which is matched here and must carry the
    // updated user identity so subsequent RPCs keep working.
    EXPECT_CALL(service_, UpdateCredential(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const UpdateCredentialRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->username(), "username");
            EXPECT_TRUE(request->description().empty());
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest* request, ConnectResponse*) {
            EXPECT_EQ(request->client_info().user(), "username");
            return ::grpc::Status{};
        });

    auto status = client->UpdatePassword(milvus::UpdatePasswordRequest()
                                             .WithUserName("username")
                                             .WithOldPassword("old")
                                             .WithNewPassword("new")
                                             .WithResetConnection(true));
    EXPECT_TRUE(status.IsOk());
}
