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

milvus::Status
DoShowPartitions(testing::StrictMock<milvus::MilvusMockedService>& service_, milvus::MilvusClientPtr& client_,
                 milvus::PartitionsInfo partitions_expected, bool only_show_loaded,
                 milvus::PartitionsInfo& partitions_received) {
    const std::string collection_name{"Foo"};
    partitions_received.clear();
    EXPECT_CALL(service_, ShowPartitions(_,
                                         AllOf(Property(&ShowPartitionsRequest::collection_name, collection_name),
                                               Property(&ShowPartitionsRequest::partition_names_size, 0)),
                                         _))
        .WillOnce([&partitions_expected](::grpc::ServerContext*, const ShowPartitionsRequest*,
                                         ShowPartitionsResponse* response) {
            for (const auto& partition : partitions_expected) {
                response->add_partition_names(partition.Name());
                response->add_partitionids(partition.Id());
                response->add_created_timestamps(partition.CreatedUtcTimestamp());
            }
            return ::grpc::Status{};
        });

    return client_->ListPartitions(collection_name, partitions_received, only_show_loaded);
}

TEST_F(MilvusMockedTest, ShowPartitions) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    milvus::PartitionsInfo partitions_expected{{"Part1", 1, 0}, {"Part2", 2, 0}};
    milvus::PartitionsInfo partitions_received;
    auto status = DoShowPartitions(service_, client_, partitions_expected, true, partitions_received);
    EXPECT_TRUE(status.IsOk());
    EXPECT_THAT(partitions_received, ElementsAreArray(partitions_expected));

    status = DoShowPartitions(service_, client_, partitions_expected, false, partitions_received);
    EXPECT_TRUE(status.IsOk());
    EXPECT_THAT(partitions_received, ElementsAreArray(partitions_expected));
}
