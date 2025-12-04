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
using ::milvus::proto::milvus::UpdateResourceGroupsRequest;
using ::testing::_;
using ::testing::Property;

void
CompareResourceGroupConfig(const milvus::ResourceGroupConfig& config,
                           const ::milvus::proto::rg::ResourceGroupConfig& rpc_config) {
    EXPECT_EQ(rpc_config.requests().node_num(), static_cast<int32_t>(config.Requests()));
    EXPECT_EQ(rpc_config.limits().node_num(), static_cast<int32_t>(config.Limits()));

    const auto& transfer_from = config.TransferFromGroups();
    const auto& rpc_from = rpc_config.transfer_from();
    EXPECT_EQ(rpc_from.size(), transfer_from.size());
    for (auto i = 0; i < rpc_from.size(); i++) {
        EXPECT_TRUE(transfer_from.find(rpc_from.at(i).resource_group()) != transfer_from.end());
    }

    const auto& transfer_to = config.TransferToGroups();
    const auto& rpc_to = rpc_config.transfer_to();
    EXPECT_EQ(rpc_to.size(), transfer_to.size());
    for (auto i = 0; i < rpc_to.size(); i++) {
        EXPECT_TRUE(transfer_to.find(rpc_to.at(i).resource_group()) != transfer_to.end());
    }

    const auto& node_filters = config.NodeFilters();
    const auto& rpc_filters = rpc_config.node_filter().node_labels();
    EXPECT_EQ(rpc_filters.size(), node_filters.size());
    for (auto i = 0; i < rpc_filters.size(); i++) {
        EXPECT_TRUE(node_filters.find(rpc_filters.at(i).key()) != node_filters.end());
        EXPECT_EQ(rpc_filters.at(i).value(), node_filters.at(rpc_filters.at(i).key()));
    }
}

TEST_F(MilvusMockedTest, UpdateResourceGroups) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::unordered_map<std::string, milvus::ResourceGroupConfig> groups;
    for (uint32_t i = 0; i < 5; i++) {
        milvus::ResourceGroupConfig config;
        config.SetRequests(i);
        config.SetLimits(i);
        config.AddTrnasferFromGroup("A" + std::to_string(i));
        config.AddTrnasferToGroup("B" + std::to_string(i));
        config.AddNodeFilter("CPU", "32c");
        config.AddNodeFilter("MEM", "16G");
        groups.insert(std::make_pair("Foo" + std::to_string(i), config));
    }

    EXPECT_CALL(service_, UpdateResourceGroups(_, _, _))
        .WillOnce([&groups](::grpc::ServerContext*, const UpdateResourceGroupsRequest* request,
                            ::milvus::proto::common::Status*) {
            const auto& rpc_groups = request->resource_groups();
            EXPECT_EQ(rpc_groups.size(), groups.size());
            for (const auto& pair : rpc_groups) {
                auto found = groups.find(pair.first);
                EXPECT_TRUE(found != groups.end());
                CompareResourceGroupConfig(found->second, pair.second);
            }
            return ::grpc::Status{};
        });

    auto status = client_->UpdateResourceGroups(groups);
    EXPECT_TRUE(status.IsOk());
}
