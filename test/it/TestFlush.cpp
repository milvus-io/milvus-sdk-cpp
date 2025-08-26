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

#include <chrono>

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::FlushRequest;
using ::milvus::proto::milvus::FlushResponse;
using ::milvus::proto::schema::LongArray;
using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(UnconnectMilvusMockedTest, FlushWithoutConnect) {
    const std::vector<std::string> collections{"c1", "c2"};
    const auto progress_monitor = ::milvus::ProgressMonitor::NoWait();
    auto status = client_->Flush(collections, progress_monitor);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, FlushInstantly) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::vector<std::string> collections{"c1", "c2"};
    const auto progress_monitor = ::milvus::ProgressMonitor::NoWait();

    EXPECT_CALL(service_, Flush(_, AllOf(Property(&FlushRequest::collection_names_size, collections.size())), _))
        .WillOnce(
            [](::grpc::ServerContext*, const FlushRequest*, FlushResponse* response) { return ::grpc::Status{}; });

    auto status = client_->Flush(collections, progress_monitor);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, FlushFailure) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::vector<std::string> collections{"c1", "c2"};
    const auto progress_monitor = ::milvus::ProgressMonitor::NoWait();

    EXPECT_CALL(service_, Flush(_, AllOf(Property(&FlushRequest::collection_names_size, collections.size())), _))
        .WillOnce([](::grpc::ServerContext*, const FlushRequest*, FlushResponse* response) {
            response->mutable_status()->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
            return ::grpc::Status{};
        });

    auto status = client_->Flush(collections, progress_monitor);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}
