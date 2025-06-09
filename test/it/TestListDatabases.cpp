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
using ::milvus::proto::milvus::ListDatabasesRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, ListDatabasesSuccess) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    EXPECT_CALL(service_, ListDatabases(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListDatabasesRequest* request,
                     ::milvus::proto::milvus::ListDatabasesResponse* response) {
            response->add_db_names("Foo");
            return ::grpc::Status{};
        });
    std::vector<std::string> names;
    auto status = client_->ListDatabases(names);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(names.size(), 1);
    EXPECT_EQ(names.at(0), "Foo");
}

TEST_F(MilvusMockedTest, ListDatabasesWithoutConnect) {
    std::vector<std::string> names;
    auto status = client_->ListDatabases(names);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, ListDatabasesFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    auto error_code = milvus::proto::common::ErrorCode::UnexpectedError;
    EXPECT_CALL(service_, ListDatabases(_, _, _))
        .WillOnce([error_code](::grpc::ServerContext*, const ListDatabasesRequest* request,
                               ::milvus::proto::milvus::ListDatabasesResponse* response) {
            response->add_db_names("Foo");
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
        });
    std::vector<std::string> names;
    auto status = client_->ListDatabases(names);

    EXPECT_FALSE(status.IsOk());
    EXPECT_TRUE(names.empty());
}