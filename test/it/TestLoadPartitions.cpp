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
using ::milvus::proto::milvus::LoadPartitionsRequest;
using ::milvus::proto::milvus::ShowPartitionsRequest;
using ::milvus::proto::milvus::ShowPartitionsResponse;
using ::milvus::proto::milvus::ShowType;
using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, LoadPartitionsInstantly) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    const std::vector<std::string> partitions{"part1", "part2"};
    const auto progress_monitor = ::milvus::ProgressMonitor::NoWait();

    EXPECT_CALL(service_,
                LoadPartitions(_,
                               AllOf(Property(&LoadPartitionsRequest::collection_name, collection),

                                     Property(&LoadPartitionsRequest::partition_names_size, partitions.size())),
                               _))
        .WillOnce([](::grpc::ServerContext*, const LoadPartitionsRequest*, milvus::proto::common::Status* response) {
            return ::grpc::Status{};
        });

    auto status = client_->LoadPartitions(collection, partitions, progress_monitor);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, LoadPartitionsFailure) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    const std::vector<std::string> partitions{"part1", "part2"};
    const ::milvus::ProgressMonitor progress_monitor{5};

    EXPECT_CALL(service_,
                LoadPartitions(_,
                               AllOf(Property(&LoadPartitionsRequest::collection_name, collection),

                                     Property(&LoadPartitionsRequest::partition_names_size, partitions.size())),
                               _))
        .WillOnce([](::grpc::ServerContext*, const LoadPartitionsRequest*, milvus::proto::common::Status* response) {
            response->set_error_code(::milvus::proto::common::ErrorCode::UnexpectedError);
            return ::grpc::Status{};
        });

    auto status = client_->LoadPartitions(collection, partitions, progress_monitor);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(MilvusMockedTest, LoadPartitionsWithQueryStatusSuccess) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    const std::vector<std::string> partitions{"part1", "part2", "part3", "part4", "part5"};
    ::milvus::ProgressMonitor progress_monitor{10};
    std::vector<milvus::Progress> progresses{};
    progress_monitor.SetCheckInterval(1);
    progress_monitor.SetCallbackFunc([&progresses](milvus::Progress& progress) { progresses.emplace_back(progress); });

    EXPECT_CALL(service_,
                LoadPartitions(_,
                               AllOf(Property(&LoadPartitionsRequest::collection_name, collection),

                                     Property(&LoadPartitionsRequest::partition_names_size, partitions.size())),
                               _))
        .WillOnce([](::grpc::ServerContext*, const LoadPartitionsRequest*, milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    int show_partitions_called = 0;
    EXPECT_CALL(service_,
                ShowPartitions(_,
                               AllOf(Property(&ShowPartitionsRequest::collection_name, collection),

                                     Property(&ShowPartitionsRequest::partition_names_size, partitions.size())),
                               _))
        .Times(10)
        .WillRepeatedly([&show_partitions_called, &partitions](::grpc::ServerContext*, const ShowPartitionsRequest*,
                                                               ShowPartitionsResponse* response) {
            ++show_partitions_called;
            int index = 0;
            for (const auto& partition : partitions) {
                ++index;
                response->add_partition_names(partition);
                response->add_partitionids(0);
                response->add_created_timestamps(0);

                // time 0: 10%, 20%, 30%, 40%, 50%      -->{0, 5}
                // time 1: 20%, 40%, 60%, 80%, 100%     -->{1, 5}
                // time 2: 30%, 60%, 90%, 100%, 100%    -->{2, 5}
                // time 3: 40%, 80%, 100%, 100%, 100%   -->{3, 5}
                // time 4: 50%, 100%, 100%, 100%, 100%  -->{4, 5}
                // time 5: 60%, 100%, 100%, 100%, 100%  -->{4, 5}
                // time 6: 70%, 100%, 100%, 100%, 100%  -->{4, 5}
                // time 7: 80%, 100%, 100%, 100%, 100%  -->{4, 5}
                // time 8: 90%, 100%, 100%, 100%, 100%  -->{4, 5}
                // time 9: 100%, 100%, 100%, 100%, 100% -->{5, 5}
                response->add_inmemory_percentages(std::min(index * 10 * show_partitions_called, 100));
            }
            return ::grpc::Status{};
        });

    auto status = client_->LoadPartitions(collection, partitions, progress_monitor);

    std::vector<milvus::Progress> progresses_expected{{0, 5}, {1, 5}, {2, 5}, {3, 5}, {4, 5},
                                                      {4, 5}, {4, 5}, {4, 5}, {4, 5}, {5, 5}};

    EXPECT_THAT(progresses, ElementsAreArray(progresses_expected));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, LoadPartitionsWithQueryStatusOomFailure) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    const std::vector<std::string> partitions{"part1", "part2"};
    ::milvus::ProgressMonitor progress_monitor{10};
    progress_monitor.SetCheckInterval(1);

    EXPECT_CALL(service_,
                LoadPartitions(_,
                               AllOf(Property(&LoadPartitionsRequest::collection_name, collection),

                                     Property(&LoadPartitionsRequest::partition_names_size, partitions.size())),
                               _))
        .WillOnce([](::grpc::ServerContext*, const LoadPartitionsRequest*, milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_,
                ShowPartitions(_,
                               AllOf(Property(&ShowPartitionsRequest::collection_name, collection),

                                     Property(&ShowPartitionsRequest::partition_names_size, partitions.size())),
                               _))
        .Times(1)
        .WillRepeatedly(
            [&partitions](::grpc::ServerContext*, const ShowPartitionsRequest*, ShowPartitionsResponse* response) {
                for (const auto& partition : partitions) {
                    response->add_partition_names(partition);
                    response->add_partitionids(0);
                    response->add_created_timestamps(0);
                    response->add_inmemory_percentages(10);
                }
                response->mutable_status()->set_error_code(::milvus::proto::common::ErrorCode::OutOfMemory);
                return ::grpc::Status{};
            });

    auto status = client_->LoadPartitions(collection, partitions, progress_monitor);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(MilvusMockedTest, LoadPartitionsWithQueryStatusTimeout) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    const std::vector<std::string> partitions{"part1", "part2"};
    ::milvus::ProgressMonitor progress_monitor{1};
    progress_monitor.SetCheckInterval(110);

    EXPECT_CALL(service_,
                LoadPartitions(_,
                               AllOf(Property(&LoadPartitionsRequest::collection_name, collection),

                                     Property(&LoadPartitionsRequest::partition_names_size, partitions.size())),
                               _))
        .WillOnce([](::grpc::ServerContext*, const LoadPartitionsRequest*, milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_,
                ShowPartitions(_,
                               AllOf(Property(&ShowPartitionsRequest::collection_name, collection),

                                     Property(&ShowPartitionsRequest::partition_names_size, partitions.size())),
                               _))
        .Times(10)
        .WillRepeatedly(
            [&partitions](::grpc::ServerContext*, const ShowPartitionsRequest*, ShowPartitionsResponse* response) {
                for (const auto& partition : partitions) {
                    response->add_partition_names(partition);
                    response->add_partitionids(0);
                    response->add_created_timestamps(0);
                    response->add_inmemory_percentages(0);
                }
                return ::grpc::Status{};
            });

    auto started = std::chrono::steady_clock::now();
    auto status = client_->LoadPartitions(collection, partitions, progress_monitor);
    auto finished = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finished - started).count();

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::TIMEOUT);
}
