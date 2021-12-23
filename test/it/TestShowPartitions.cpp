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
using ::milvus::proto::milvus::ShowPartitionsRequest;
using ::milvus::proto::milvus::ShowPartitionsResponse;
using ::milvus::proto::milvus::ShowType;
using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, ShowPartitionsQueryAll) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    ::milvus::PartitionsInfo partitions_expected{{"Part1", 1, 0, 0}, {"Part2", 2, 0, 0}};

    EXPECT_CALL(service_, ShowPartitions(_,
                                         AllOf(Property(&ShowPartitionsRequest::collection_name, collection),

                                               Property(&ShowPartitionsRequest::partition_names_size, 0),
                                               Property(&ShowPartitionsRequest::type, ShowType::All)),
                                         _))
        .WillOnce([&partitions_expected](::grpc::ServerContext*, const ShowPartitionsRequest*,
                                         ShowPartitionsResponse* response) {
            for (const auto& partition : partitions_expected) {
                response->add_partition_names(partition.Name());
                response->add_partitionids(partition.Id());
                response->add_created_timestamps(partition.CreatedUtcTimestamp());
                response->add_inmemory_percentages(partition.InMemoryPercentage());
            }
            return ::grpc::Status{};
        });

    ::milvus::PartitionsInfo partitions_received{};

    auto status = client_->ShowPartitions(collection, std::vector<std::string>{}, partitions_received);

    EXPECT_TRUE(status.IsOk());
    EXPECT_THAT(partitions_received, ElementsAreArray(partitions_expected));
}

TEST_F(MilvusMockedTest, ShowPartitionsQueryLoadingPercent) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    ::milvus::PartitionsInfo partitions_expected{{"Part1", 1, 0, 100}, {"Part2", 2, 0, 30}};
    std::vector<std::string> partition_names{{"Part1"}, {"Part2"}};

    EXPECT_CALL(
        service_,
        ShowPartitions(_,
                       AllOf(Property(&ShowPartitionsRequest::collection_name, collection),
                             Property(&ShowPartitionsRequest::partition_names, ElementsAreArray(partition_names)),
                             Property(&ShowPartitionsRequest::type, ShowType::InMemory)),
                       _))
        .WillOnce([&partitions_expected](::grpc::ServerContext*, const ShowPartitionsRequest*,
                                         ShowPartitionsResponse* response) {
            for (const auto& partition : partitions_expected) {
                response->add_partition_names(partition.Name());
                response->add_partitionids(partition.Id());
                response->add_created_timestamps(partition.CreatedUtcTimestamp());
                response->add_inmemory_percentages(partition.InMemoryPercentage());
            }
            return ::grpc::Status{};
        });

    ::milvus::PartitionsInfo partitions_received{};

    auto status = client_->ShowPartitions(collection, partition_names, partitions_received);

    EXPECT_TRUE(status.IsOk());
    EXPECT_THAT(partitions_received, ElementsAreArray(partitions_expected));
}
