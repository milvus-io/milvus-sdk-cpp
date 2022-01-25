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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::FlushRequest;
using ::milvus::proto::milvus::FlushResponse;
using ::milvus::proto::milvus::GetPartitionStatisticsRequest;
using ::milvus::proto::milvus::GetPartitionStatisticsResponse;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Property;

TEST_F(MilvusMockedTest, GetPartitionStatisticsWithoutConnection) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    std::string collection = "Foo";
    std::string partition = "Bar";

    milvus::PartitionStat partition_stat;
    auto status = client_->GetPartitionStatistics(collection, partition, partition_stat);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, GetPartitionStatisticsInstantly) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection = "Foo";
    std::string partition = "Bar";

    EXPECT_CALL(service_,
                GetPartitionStatistics(_,
                                       AllOf(Property(&GetPartitionStatisticsRequest::collection_name, collection),
                                             Property(&GetPartitionStatisticsRequest::partition_name, partition)),
                                       _))
        .WillOnce(
            [](::grpc::ServerContext*, const GetPartitionStatisticsRequest*, GetPartitionStatisticsResponse* response) {
                auto* stats = response->add_stats();
                stats->set_key("row_count");
                stats->set_value("1000");
                return ::grpc::Status{};
            });
    milvus::PartitionStat partition_stat;
    auto status =
        client_->GetPartitionStatistics(collection, partition, partition_stat, milvus::ProgressMonitor::NoWait());

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(partition_stat.RowCount(), 1000);
}

TEST_F(MilvusMockedTest, GetPartitionStatisticsWithFlushFailure) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection = "Foo";
    std::string partition = "Bar";

    EXPECT_CALL(service_, Flush(_, AllOf(Property(&FlushRequest::collection_names, ElementsAre(collection))), _))
        .WillOnce([](::grpc::ServerContext*, const FlushRequest*, FlushResponse* response) {
            response->mutable_status()->set_error_code(::milvus::proto::common::ErrorCode::UnexpectedError);
            return ::grpc::Status{};
        });

    milvus::PartitionStat partition_stat;
    auto status = client_->GetPartitionStatistics(collection, partition, partition_stat, milvus::ProgressMonitor{1});

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::SERVER_FAILED);
}

TEST_F(MilvusMockedTest, GetPartitionStatisticsWithFlushInstantly) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection = "Foo";
    std::string partition = "Bar";

    EXPECT_CALL(service_, Flush(_, AllOf(Property(&FlushRequest::collection_names, ElementsAre(collection))), _))
        .WillOnce(
            [](::grpc::ServerContext*, const FlushRequest*, FlushResponse* response) { return ::grpc::Status{}; });

    EXPECT_CALL(service_,
                GetPartitionStatistics(_,
                                       AllOf(Property(&GetPartitionStatisticsRequest::collection_name, collection),
                                             Property(&GetPartitionStatisticsRequest::partition_name, partition)),
                                       _))
        .WillOnce(
            [](::grpc::ServerContext*, const GetPartitionStatisticsRequest*, GetPartitionStatisticsResponse* response) {
                auto* stats = response->add_stats();
                stats->set_key("row_count");
                stats->set_value("1000");
                return ::grpc::Status{};
            });

    milvus::PartitionStat partition_stat;
    auto status = client_->GetPartitionStatistics(collection, partition, partition_stat, milvus::ProgressMonitor{1});

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(partition_stat.RowCount(), 1000);
}
