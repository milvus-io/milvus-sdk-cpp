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
using ::milvus::proto::milvus::LoadBalanceRequest;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, LoadBalanceFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    int64_t src_node = 1000;
    std::vector<int64_t> dst_nodes = {2000, 2001, 2002};
    std::vector<int64_t> segments = {3000, 3001, 3002};

    EXPECT_CALL(service_,
                LoadBalance(_,
                            AllOf(Property(&LoadBalanceRequest::src_nodeid, src_node),
                                  Property(&LoadBalanceRequest::dst_nodeids, ElementsAreArray(dst_nodes)),
                                  Property(&LoadBalanceRequest::sealed_segmentids, ElementsAreArray(segments))),
                            _))
        .WillOnce([](::grpc::ServerContext*, const LoadBalanceRequest*, milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    auto status = client_->LoadBalance(src_node, dst_nodes, segments);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, LoadBalanceWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    int64_t src_node = 1000;
    std::vector<int64_t> dst_nodes = {2000, 2001, 2002};
    std::vector<int64_t> segments = {3000, 3001, 3002};

    auto status = client_->LoadBalance(src_node, dst_nodes, segments);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::NOT_CONNECTED);
}