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
using ::milvus::proto::milvus::CreateDatabaseRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, CreateDatabaseSuccess) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    EXPECT_CALL(service_, CreateDatabase(_, Property(&CreateDatabaseRequest::db_name, "Foo"), _))
        .WillOnce([](::grpc::ServerContext*, const CreateDatabaseRequest* request, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });
    std::unordered_map<std::string, std::string> properties{};
    auto status = client_->CreateDatabase("Foo", properties);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, CreateDatabaseWithoutConnect) {
    std::unordered_map<std::string, std::string> properties{};
    auto status = client_->CreateDatabase("Foo", properties);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, CreateDatabaseFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    auto error_code = milvus::proto::common::ErrorCode::UnexpectedError;
    EXPECT_CALL(service_, CreateDatabase(_, Property(&CreateDatabaseRequest::db_name, "Foo"), _))
        .WillOnce([error_code](::grpc::ServerContext*, const CreateDatabaseRequest* request,
                               ::milvus::proto::common::Status* status) {
            status->set_code(error_code);
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
        });
    std::unordered_map<std::string, std::string> properties{};
    auto status = client_->CreateDatabase("Foo", properties);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}