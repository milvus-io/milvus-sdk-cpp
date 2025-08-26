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
