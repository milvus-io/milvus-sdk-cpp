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
using ::milvus::proto::milvus::ReleasePartitionsRequest;
using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, ReleasePartitionsNotConnected) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    const std::string collection{"Foo"};
    const std::vector<std::string> partitions{"part1", "part2"};

    auto status = client_->ReleasePartitions(collection, partitions);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, ReleasePartitionsFailure) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    const std::vector<std::string> partitions{"part1", "part2"};

    EXPECT_CALL(service_,
                ReleasePartitions(_,
                                  AllOf(Property(&ReleasePartitionsRequest::collection_name, collection),

                                        Property(&ReleasePartitionsRequest::partition_names_size, partitions.size())),
                                  _))
        .WillOnce([](::grpc::ServerContext*, const ReleasePartitionsRequest*, milvus::proto::common::Status* response) {
            response->set_error_code(::milvus::proto::common::ErrorCode::UnexpectedError);
            return ::grpc::Status{};
        });

    auto status = client_->ReleasePartitions(collection, partitions);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(MilvusMockedTest, ReleasePartitionsSuccess) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    const std::vector<std::string> partitions{"part1", "part2"};

    EXPECT_CALL(service_,
                ReleasePartitions(_,
                                  AllOf(Property(&ReleasePartitionsRequest::collection_name, collection),

                                        Property(&ReleasePartitionsRequest::partition_names_size, partitions.size())),
                                  _))
        .WillOnce([](::grpc::ServerContext*, const ReleasePartitionsRequest*, milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    auto status = client_->ReleasePartitions(collection, partitions);
    EXPECT_TRUE(status.IsOk());
}
