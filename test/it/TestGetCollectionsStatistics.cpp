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
using ::milvus::proto::milvus::GetCollectionStatisticsRequest;
using ::milvus::proto::milvus::GetCollectionStatisticsResponse;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Property;

TEST_F(MilvusMockedTest, GetCollectionStatisticsInstantly) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection = "Foo";
    milvus::CollectionStat collection_stat;

    EXPECT_CALL(service_, GetCollectionStatistics(
                              _, AllOf(Property(&GetCollectionStatisticsRequest::collection_name, collection)), _))
        .WillOnce([](::grpc::ServerContext*, const GetCollectionStatisticsRequest*,
                     GetCollectionStatisticsResponse* response) {
            auto* stats = response->add_stats();
            stats->set_key("row_count");
            stats->set_value("1000");
            return grpc::Status{};
        });

    auto status = client_->GetCollectionStatistics(collection, collection_stat, milvus::ProgressMonitor::NoWait());
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(collection_stat.RowCount(), 1000);
}

TEST_F(MilvusMockedTest, GetCollectionStatisticsWithFlushInstantly) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection = "Foo";
    milvus::CollectionStat collection_stat;

    EXPECT_CALL(service_, Flush(_, AllOf(Property(&FlushRequest::collection_names, ElementsAre(collection))), _))
        .WillOnce(
            [](::grpc::ServerContext*, const FlushRequest*, FlushResponse* response) { return ::grpc::Status{}; });

    EXPECT_CALL(service_, GetCollectionStatistics(
                              _, AllOf(Property(&GetCollectionStatisticsRequest::collection_name, collection)), _))
        .WillOnce([](::grpc::ServerContext*, const GetCollectionStatisticsRequest*,
                     GetCollectionStatisticsResponse* response) {
            auto* stats = response->add_stats();
            stats->set_key("row_count");
            stats->set_value("1000");
            return grpc::Status{};
        });

    auto status = client_->GetCollectionStatistics(collection, collection_stat, milvus::ProgressMonitor{1});

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(collection_stat.RowCount(), 1000);
}

TEST_F(MilvusMockedTest, GetCollectionStatisticsWithFlushFailure) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection = "Foo";
    milvus::CollectionStat collection_stat;

    EXPECT_CALL(service_, Flush(_, AllOf(Property(&FlushRequest::collection_names, ElementsAre(collection))), _))
        .WillOnce([](::grpc::ServerContext*, const FlushRequest*, FlushResponse* response) {
            response->mutable_status()->set_error_code(::milvus::proto::common::ErrorCode::UnexpectedError);
            return ::grpc::Status{};
        });

    auto status = client_->GetCollectionStatistics(collection, collection_stat, milvus::ProgressMonitor{1});
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::SERVER_FAILED);
}