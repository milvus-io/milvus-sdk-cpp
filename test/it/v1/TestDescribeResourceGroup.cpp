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
#include "utils/TypeUtils.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::DescribeResourceGroupRequest;
using ::milvus::proto::milvus::DescribeResourceGroupResponse;
using ::testing::_;
using ::testing::Property;

void
CompareResourceGroupConfig(const milvus::ResourceGroupConfig& config_1, const milvus::ResourceGroupConfig& config_2) {
    const auto& from_1 = config_1.TransferFromGroups();
    const auto& from_2 = config_2.TransferFromGroups();
    EXPECT_EQ(from_1.size(), from_2.size());
    for (const auto& name : from_1) {
        EXPECT_TRUE(from_2.find(name) != from_2.end());
    }

    const auto& to_1 = config_1.TransferToGroups();
    const auto& to_2 = config_2.TransferToGroups();
    EXPECT_EQ(to_1.size(), to_2.size());
    for (const auto& name : to_1) {
        EXPECT_TRUE(to_2.find(name) != to_2.end());
    }

    const auto& filters_1 = config_1.NodeFilters();
    const auto& filters_2 = config_2.NodeFilters();
    EXPECT_EQ(filters_1.size(), filters_2.size());
    for (const auto& pair : filters_1) {
        EXPECT_TRUE(filters_2.find(pair.first) != filters_2.end());
        EXPECT_EQ(filters_1.at(pair.first), filters_2.at(pair.first));
    }
}

TEST_F(MilvusMockedTest, DescribeResourceGroup) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string group_name = "Foo";
    uint32_t expected_capacity = 5;
    uint32_t expected_num_available_node = 2;
    std::unordered_map<std::string, uint32_t> expected_num_loaded_replica;
    expected_num_loaded_replica.insert(std::make_pair("A", 4));
    expected_num_loaded_replica.insert(std::make_pair("A", 5));
    std::unordered_map<std::string, uint32_t> expected_num_outgoing_node;
    expected_num_outgoing_node.insert(std::make_pair("C", 6));
    expected_num_outgoing_node.insert(std::make_pair("D", 7));
    std::unordered_map<std::string, uint32_t> expected_num_incoming_node;
    expected_num_incoming_node.insert(std::make_pair("E", 8));
    expected_num_incoming_node.insert(std::make_pair("F", 9));

    milvus::ResourceGroupConfig expected_config;
    expected_config.SetRequests(3);
    expected_config.SetLimits(5);
    expected_config.AddTrnasferFromGroup("A");
    expected_config.AddTrnasferToGroup("B");
    expected_config.AddNodeFilter("cu", "32c");

    std::vector<milvus::NodeInfo> expected_nodes;
    expected_nodes.emplace_back(123, "127.0.0.1", "localhost");
    expected_nodes.emplace_back(456, "http://localhost", "server");

    EXPECT_CALL(service_,
                DescribeResourceGroup(_, Property(&DescribeResourceGroupRequest::resource_group, group_name), _))
        .WillOnce([&group_name, &expected_capacity, &expected_num_available_node, &expected_num_loaded_replica,
                   &expected_num_outgoing_node, &expected_num_incoming_node, &expected_config,
                   &expected_nodes](::grpc::ServerContext*, const DescribeResourceGroupRequest*,
                                    DescribeResourceGroupResponse* response) {
            auto rpc_desc = response->mutable_resource_group();
            rpc_desc->set_name(group_name);
            rpc_desc->set_capacity(expected_capacity);
            rpc_desc->set_num_available_node(expected_num_available_node);

            for (auto& pair : expected_num_loaded_replica) {
                rpc_desc->mutable_num_loaded_replica()->insert(std::make_pair(pair.first, pair.second));
            }
            for (auto& pair : expected_num_outgoing_node) {
                rpc_desc->mutable_num_outgoing_node()->insert(std::make_pair(pair.first, pair.second));
            }
            for (auto& pair : expected_num_incoming_node) {
                rpc_desc->mutable_num_incoming_node()->insert(std::make_pair(pair.first, pair.second));
            }

            auto rpc_config = new ::milvus::proto::rg::ResourceGroupConfig{};
            milvus::ConvertResourceGroupConfig(expected_config, rpc_config);
            rpc_desc->set_allocated_config(rpc_config);

            for (auto& node : expected_nodes) {
                auto rpc_node = rpc_desc->mutable_nodes()->Add();
                rpc_node->set_node_id(node.id_);
                rpc_node->set_address(node.address_);
                rpc_node->set_hostname(node.hostname_);
            }
            return ::grpc::Status{};
        });

    milvus::ResourceGroupDesc desc;
    auto status = client_->DescribeResourceGroup(group_name, desc);
    EXPECT_TRUE(status.IsOk());

    EXPECT_EQ(desc.Name(), group_name);
    EXPECT_EQ(desc.Capacity(), expected_capacity);
    EXPECT_EQ(desc.AvailableNodesNum(), expected_num_available_node);

    for (const auto& pair : desc.LoadedReplicasNum()) {
        auto num = expected_num_loaded_replica.find(pair.first);
        EXPECT_TRUE(num != expected_num_loaded_replica.end());
        EXPECT_EQ(pair.second, num->second);
    }
    for (const auto& pair : desc.OutgoingNodesNum()) {
        auto num = expected_num_outgoing_node.find(pair.first);
        EXPECT_TRUE(num != expected_num_outgoing_node.end());
        EXPECT_EQ(pair.second, num->second);
    }
    for (const auto& pair : desc.IncomingNodesNum()) {
        auto num = expected_num_incoming_node.find(pair.first);
        EXPECT_TRUE(num != expected_num_incoming_node.end());
        EXPECT_EQ(pair.second, num->second);
    }

    CompareResourceGroupConfig(desc.Config(), expected_config);

    const auto& nodes = desc.Nodes();
    EXPECT_EQ(nodes.size(), expected_nodes.size());
    for (auto i = 0; i < nodes.size(); i++) {
        EXPECT_EQ(nodes[i].id_, expected_nodes[i].id_);
        EXPECT_EQ(nodes[i].address_, expected_nodes[i].address_);
        EXPECT_EQ(nodes[i].hostname_, expected_nodes[i].hostname_);
    }
}
