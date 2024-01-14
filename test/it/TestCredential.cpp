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
using ::milvus::proto::milvus::CreateCredentialRequest;
using ::milvus::proto::milvus::DeleteCredentialRequest;
using ::milvus::proto::milvus::ListCredUsersRequest;
using ::milvus::proto::milvus::UpdateCredentialRequest;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Property;

TEST_F(MilvusMockedTest, CreateCredential) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    EXPECT_CALL(service_, CreateCredential(_,
                                           AllOf(Property(&CreateCredentialRequest::username, "username"),
                                                 Property(&CreateCredentialRequest::password, "cGFzc3dvcmQ=")),
                                           _))
        .WillOnce([](::grpc::ServerContext*, const CreateCredentialRequest* request, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });
    auto status = client_->CreateCredential("username", "password");

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, CreateCredentialError) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    EXPECT_CALL(service_, CreateCredential(_,
                                           AllOf(Property(&CreateCredentialRequest::username, "username"),
                                                 Property(&CreateCredentialRequest::password, "cGFzc3dvcmQ=")),
                                           _))
        .WillOnce([](::grpc::ServerContext*, const CreateCredentialRequest* request,
                     ::milvus::proto::common::Status* status) {
            status->set_code(milvus::proto::common::ErrorCode::CreateCredentialFailure);
            return ::grpc::Status{};
        });
    auto status = client_->CreateCredential("username", "password");

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(MilvusMockedTest, UpdateCredential) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    EXPECT_CALL(service_, UpdateCredential(_,
                                           AllOf(Property(&UpdateCredentialRequest::username, "username"),
                                                 Property(&UpdateCredentialRequest::oldpassword, "b2xk"),
                                                 Property(&UpdateCredentialRequest::newpassword, "bmV3")),
                                           _))
        .WillOnce([](::grpc::ServerContext*, const UpdateCredentialRequest* request, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });
    auto status = client_->UpdateCredential("username", "old", "new");

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, DeleteCredential) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    EXPECT_CALL(service_, DeleteCredential(_, Property(&DeleteCredentialRequest::username, "username"), _))
        .WillOnce([](::grpc::ServerContext*, const DeleteCredentialRequest* request, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });
    auto status = client_->DeleteCredential("username");

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, ListCredUsers) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    EXPECT_CALL(service_, ListCredUsers(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListCredUsersRequest* request,
                     ::milvus::proto::milvus::ListCredUsersResponse* resp) {
            resp->add_usernames("foo");
            resp->add_usernames("bar");
            return ::grpc::Status{};
        });
    std::vector<std::string> users;
    auto status = client_->ListCredUsers(users);

    EXPECT_TRUE(status.IsOk());
    EXPECT_THAT(users, ElementsAre("foo", "bar"));
}
