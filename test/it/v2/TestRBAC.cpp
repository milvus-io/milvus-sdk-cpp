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

using ::milvus::StatusCode;
using ::milvus::proto::milvus::AlterRoleRequest;
using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::CreateCredentialRequest;
using ::milvus::proto::milvus::CreateRoleRequest;
using ::milvus::proto::milvus::OperatePrivilegeRequest;
using ::milvus::proto::milvus::SelectGrantRequest;
using ::milvus::proto::milvus::SelectGrantResponse;
using ::milvus::proto::milvus::SelectRoleRequest;
using ::milvus::proto::milvus::SelectRoleResponse;
using ::milvus::proto::milvus::SelectUserRequest;
using ::milvus::proto::milvus::SelectUserResponse;
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
            EXPECT_EQ(request->entity().object_name(), "*");
            EXPECT_EQ(request->entity().grantor().privilege().name(), "CreateCollection");
            EXPECT_TRUE(request->entity().db_name().empty());
            EXPECT_EQ(request->type(), ::milvus::proto::milvus::OperatePrivilegeType::Revoke);
            return ::grpc::Status{};
        });

    auto status = client->RevokePrivilege(milvus::RevokePrivilegeRequest()
                                              .WithRoleName("reader_role")
                                              .WithObjectType("Global")
                                              .WithObjectName("*")
                                              .WithPrivilege("CreateCollection"));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, UpdateUser) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, UpdateCredential(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const UpdateCredentialRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->username(), "username");
            EXPECT_EQ(request->description(), "user description");
            EXPECT_TRUE(request->oldpassword().empty());
            EXPECT_TRUE(request->newpassword().empty());
            return ::grpc::Status{};
        });

    milvus::UpdateUserRequest request;
    request.WithUserName("username").WithDescription("user description");
    auto status = client->UpdateUser(request);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, UpdateUserFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, UpdateCredential(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const UpdateCredentialRequest*, ::milvus::proto::common::Status*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::UpdateUserRequest request;
    request.WithUserName("username").WithDescription("user description");
    auto status = client->UpdateUser(request);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
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
TEST_F(UnconnectMilvusMockedTest, CreateRole) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    const std::string role_name = "Foo";

    EXPECT_CALL(service_, CreateRole(_, _, _))
        .WillOnce(
            [&role_name](::grpc::ServerContext*, const CreateRoleRequest* request, ::milvus::proto::common::Status*) {
                EXPECT_EQ(request->entity().name(), role_name);
                EXPECT_EQ(request->entity().description(), "role description");
                return ::grpc::Status{};
            });

    milvus::CreateRoleRequest request;
    request.WithRoleName(role_name).WithDescription("role description");
    auto status = client->CreateRole(request);
    EXPECT_TRUE(status.IsOk());
}
TEST_F(UnconnectMilvusMockedTest, DescribeRole) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::RoleDesc expected_desc;
    expected_desc.SetName("Foo");
    expected_desc.AddGrantItem({"a", "b", "c", "d", "e", "f"});
    expected_desc.AddGrantItem({"1", "2", "3", "4", "5", "6"});

    EXPECT_CALL(service_, SelectGrant(_, _, _))
        .WillOnce(
            [&expected_desc](::grpc::ServerContext*, const SelectGrantRequest* request, SelectGrantResponse* response) {
                EXPECT_EQ(request->entity().role().name(), expected_desc.Name());

                for (const auto& item : expected_desc.GrantItems()) {
                    auto entity = response->mutable_entities()->Add();
                    entity->mutable_object()->set_name(item.object_type_);
                    entity->set_object_name(item.object_name_);
                    entity->set_db_name(item.db_name_);
                    entity->mutable_role()->set_name(item.role_name_);
                    entity->mutable_grantor()->mutable_user()->set_name(item.grantor_name_);
                    entity->mutable_grantor()->mutable_privilege()->set_name(item.privilege_);
                }
                return ::grpc::Status{};
            });

    EXPECT_CALL(service_, SelectRole(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SelectRoleRequest* request, SelectRoleResponse* response) {
            EXPECT_EQ(request->role().name(), "Foo");
            auto result = response->mutable_results()->Add();
            result->mutable_role()->set_name("Foo");
            result->mutable_role()->set_description("role description");
            return ::grpc::Status{};
        });

    milvus::DescribeRoleRequest request;
    request.WithRoleName(expected_desc.Name());
    milvus::DescribeRoleResponse response;
    auto status = client->DescribeRole(request, response);
    EXPECT_TRUE(status.IsOk());

    const auto& desc = response.Desc();
    EXPECT_EQ(desc.Name(), expected_desc.Name());
    EXPECT_EQ(desc.Description(), "role description");
    EXPECT_EQ(desc.GrantItems().size(), expected_desc.GrantItems().size());
    for (auto i = 0; i < desc.GrantItems().size(); i++) {
        EXPECT_EQ(desc.GrantItems().at(i).object_type_, expected_desc.GrantItems().at(i).object_type_);
        EXPECT_EQ(desc.GrantItems().at(i).object_name_, expected_desc.GrantItems().at(i).object_name_);
        EXPECT_EQ(desc.GrantItems().at(i).db_name_, expected_desc.GrantItems().at(i).db_name_);
        EXPECT_EQ(desc.GrantItems().at(i).role_name_, expected_desc.GrantItems().at(i).role_name_);
        EXPECT_EQ(desc.GrantItems().at(i).grantor_name_, expected_desc.GrantItems().at(i).grantor_name_);
        EXPECT_EQ(desc.GrantItems().at(i).privilege_, expected_desc.GrantItems().at(i).privilege_);
    }
}
TEST_F(UnconnectMilvusMockedTest, AlterRole) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, AlterRole(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterRoleRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->role_name(), "role_name");
            EXPECT_EQ(request->description(), "role description");
            return ::grpc::Status{};
        });

    milvus::AlterRoleRequest request;
    request.WithRoleName("role_name").WithDescription("role description");
    auto status = client->AlterRole(request);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, AlterRoleFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, AlterRole(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterRoleRequest*, ::milvus::proto::common::Status*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::AlterRoleRequest request;
    request.WithRoleName("role_name").WithDescription("role description");
    auto status = client->AlterRole(request);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}
TEST_F(UnconnectMilvusMockedTest, CreateUser) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, CreateCredential(_,
                                           AllOf(Property(&CreateCredentialRequest::username, "username"),
                                                 Property(&CreateCredentialRequest::password, "cGFzc3dvcmQ="),
                                                 Property(&CreateCredentialRequest::description, "user description")),
                                           _))
        .WillOnce([](::grpc::ServerContext*, const CreateCredentialRequest*, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    milvus::CreateUserRequest request;
    request.WithUserName("username").WithPassword("password").WithDescription("user description");
    auto status = client->CreateUser(request);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, CreateUserError) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, CreateCredential(_,
                                           AllOf(Property(&CreateCredentialRequest::username, "username"),
                                                 Property(&CreateCredentialRequest::password, "cGFzc3dvcmQ="),
                                                 Property(&CreateCredentialRequest::description, "user description")),
                                           _))
        .WillOnce([](::grpc::ServerContext*, const CreateCredentialRequest*, ::milvus::proto::common::Status* status) {
            status->set_code(milvus::proto::common::ErrorCode::CreateCredentialFailure);
            return ::grpc::Status{};
        });

    milvus::CreateUserRequest request;
    request.WithUserName("username").WithPassword("password").WithDescription("user description");
    auto status = client->CreateUser(request);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}
TEST_F(UnconnectMilvusMockedTest, DescribeUser) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::UserDesc expected_desc;
    expected_desc.SetName("Bar");
    expected_desc.AddRole("role_1");
    expected_desc.AddRole("role_2");

    EXPECT_CALL(service_, SelectUser(_, _, _))
        .WillOnce(
            [&expected_desc](::grpc::ServerContext*, const SelectUserRequest* request, SelectUserResponse* response) {
                EXPECT_EQ(request->user().name(), expected_desc.Name());
                EXPECT_TRUE(request->include_role_info());

                auto result = response->mutable_results()->Add();
                result->mutable_user()->set_name(expected_desc.Name());
                result->set_description("user description");
                for (const auto& role : expected_desc.Roles()) {
                    result->add_roles()->set_name(role);
                }
                return ::grpc::Status{};
            });

    milvus::DescribeUserRequest request;
    request.WithUserName(expected_desc.Name());
    milvus::DescribeUserResponse response;
    auto status = client->DescribeUser(request, response);
    EXPECT_TRUE(status.IsOk());

    const auto& desc = response.Desc();
    EXPECT_EQ(desc.Name(), expected_desc.Name());
    EXPECT_EQ(desc.Description(), "user description");
    EXPECT_EQ(desc.Roles().size(), expected_desc.Roles().size());
    for (auto i = 0; i < desc.Roles().size(); i++) {
        EXPECT_EQ(desc.Roles().at(i), expected_desc.Roles().at(i));
    }
}
