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
using ::milvus::proto::milvus::TransferNodeRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, TransferNode) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string src_group = "Foo";
    const std::string target_group = "Bar";
    uint32_t num_nodes = 5;
    EXPECT_CALL(service_, TransferNode(_,
                                       AllOf(Property(&TransferNodeRequest::source_resource_group, src_group),
                                             Property(&TransferNodeRequest::target_resource_group, target_group),
                                             Property(&TransferNodeRequest::num_node, num_nodes)),
                                       _))
        .WillOnce([](::grpc::ServerContext*, const TransferNodeRequest* request, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });

    auto status = client_->TransferNode(src_group, target_group, num_nodes);
    EXPECT_TRUE(status.IsOk());
}
