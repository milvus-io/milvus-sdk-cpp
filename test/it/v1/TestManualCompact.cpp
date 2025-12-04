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
using ::milvus::proto::milvus::DescribeCollectionRequest;
using ::milvus::proto::milvus::DescribeCollectionResponse;
using ::milvus::proto::milvus::ManualCompactionRequest;
using ::milvus::proto::milvus::ManualCompactionResponse;
using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, ManualCompaction) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const int64_t compaction_id = 1;
    const std::string collection_name = "test";
    const int64_t collection_id = 9;
    const uint64_t travel_ts = 100;

    EXPECT_CALL(service_,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillOnce([&](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
            response->set_collectionid(collection_id);
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, ManualCompaction(_,
                                           AllOf(Property(&ManualCompactionRequest::collectionid, collection_id),
                                                 Property(&ManualCompactionRequest::timetravel, travel_ts)),
                                           _))
        .WillOnce([&](::grpc::ServerContext*, const ManualCompactionRequest*, ManualCompactionResponse* response) {
            response->set_compactionid(compaction_id);
            return ::grpc::Status{};
        });

    int64_t returned_compaction_id = 0;
    auto status = client_->ManualCompaction(collection_name, travel_ts, returned_compaction_id);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(returned_compaction_id, compaction_id);
}

TEST_F(UnconnectMilvusMockedTest, ManualCompactionWithoutConnect) {
    const std::string collection_name = "test";
    const uint64_t travel_ts = 100;
    int64_t returned_compaction_id = 0;
    auto status = client_->ManualCompaction(collection_name, travel_ts, returned_compaction_id);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, ManualCompactionFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection_name = "test";
    const uint64_t travel_ts = 100;

    EXPECT_CALL(service_,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillOnce([](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
        });

    int64_t returned_compaction_id = 0;
    auto status = client_->ManualCompaction(collection_name, travel_ts, returned_compaction_id);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}
