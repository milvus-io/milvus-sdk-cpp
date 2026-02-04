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
using ::milvus::proto::milvus::CreateResourceGroupRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, CreateResourceGroup) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string group_name = "Foo";

    milvus::ResourceGroupConfig config;
    config.SetRequests(3);
    config.SetLimits(5);
    config.AddTrnasferFromGroup("A");
    config.AddTrnasferToGroup("B");
    config.AddNodeFilter("cu", "32c");

    EXPECT_CALL(service_, CreateResourceGroup(_, Property(&CreateResourceGroupRequest::resource_group, group_name), _))
        .WillOnce([&config](::grpc::ServerContext*, const CreateResourceGroupRequest* request,
                            ::milvus::proto::common::Status*) {
            auto rpc_config = request->config();
            EXPECT_EQ(rpc_config.requests().node_num(), static_cast<int32_t>(config.Requests()));
            EXPECT_EQ(rpc_config.limits().node_num(), static_cast<int32_t>(config.Limits()));

            const auto& transfer_from = config.TransferFromGroups();
            const auto& rpc_from = rpc_config.transfer_from();
            EXPECT_EQ(rpc_from.size(), transfer_from.size());
            for (const auto& item : rpc_from) {
                EXPECT_TRUE(transfer_from.find(item.resource_group()) != transfer_from.end());
            }

            const auto& transfer_to = config.TransferToGroups();
            const auto& rpc_to = rpc_config.transfer_to();
            EXPECT_EQ(rpc_to.size(), transfer_to.size());
            for (const auto& item : rpc_to) {
                EXPECT_TRUE(transfer_to.find(item.resource_group()) != transfer_to.end());
            }

            const auto& node_filters = config.NodeFilters();
            const auto& rpc_filters = rpc_config.node_filter().node_labels();
            EXPECT_EQ(rpc_filters.size(), node_filters.size());
            for (const auto& rpc_filter : rpc_filters) {
                EXPECT_TRUE(node_filters.find(rpc_filter.key()) != node_filters.end());
                EXPECT_EQ(rpc_filter.value(), node_filters.at(rpc_filter.key()));
            }

            return ::grpc::Status{};
        });

    auto status = client_->CreateResourceGroup(group_name, config);
    EXPECT_TRUE(status.IsOk());
}
