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

#include "../mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::GetLoadingProgressRequest;
using ::milvus::proto::milvus::GetLoadingProgressResponse;
using ::milvus::proto::milvus::LoadCollectionRequest;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Property;

TEST_F(MilvusMockedTest, LoadCollectionInstantly) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "Foo";
    const auto progress_monitor = ::milvus::ProgressMonitor::NoWait();

    EXPECT_CALL(service_, LoadCollection(_,
                                         AllOf(Property(&LoadCollectionRequest::collection_name, collection_name),
                                               Property(&LoadCollectionRequest::replica_number, 1)),
                                         _))
        .WillOnce([](::grpc::ServerContext*, const LoadCollectionRequest*, milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    auto status = client_->LoadCollection(collection_name, 1, progress_monitor);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, LoadCollectionWithProgress) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "Foo";
    ::milvus::ProgressMonitor progress_monitor{10};
    std::vector<milvus::Progress> progresses{};
    progress_monitor.SetCheckInterval(1);
    progress_monitor.SetCallbackFunc([&progresses](milvus::Progress& progress) { progresses.emplace_back(progress); });

    EXPECT_CALL(service_, LoadCollection(_,
                                         AllOf(Property(&LoadCollectionRequest::collection_name, collection_name),
                                               Property(&LoadCollectionRequest::replica_number, 2)),
                                         _))
        .WillOnce([](::grpc::ServerContext*, const LoadCollectionRequest*, milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    int called_times{0};
    EXPECT_CALL(service_, GetLoadingProgress(_, _, _))
        .Times(10)
        .WillRepeatedly(
            [&](::grpc::ServerContext*, const GetLoadingProgressRequest*, GetLoadingProgressResponse* response) {
                response->set_progress((++called_times) * 10);
                return ::grpc::Status{};
            });

    auto status = client_->LoadCollection(collection_name, 2, progress_monitor);
    EXPECT_TRUE(status.IsOk());
}