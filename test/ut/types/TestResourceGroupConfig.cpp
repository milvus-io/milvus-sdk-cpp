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

#include "milvus/MilvusClientV2.h"

class ResourceGroupConfigTest : public ::testing::Test {};

TEST_F(ResourceGroupConfigTest, DefaultValues) {
    milvus::ResourceGroupConfig config;
    EXPECT_EQ(config.Requests(), 0u);
    EXPECT_EQ(config.Limits(), 0u);
    EXPECT_TRUE(config.TransferFromGroups().empty());
    EXPECT_TRUE(config.TransferToGroups().empty());
    EXPECT_TRUE(config.NodeFilters().empty());
}

TEST_F(ResourceGroupConfigTest, RequestsGetterSetter) {
    milvus::ResourceGroupConfig config;
    config.SetRequests(5);
    EXPECT_EQ(config.Requests(), 5u);
}

TEST_F(ResourceGroupConfigTest, LimitsGetterSetter) {
    milvus::ResourceGroupConfig config;
    config.SetLimits(10);
    EXPECT_EQ(config.Limits(), 10u);
}

TEST_F(ResourceGroupConfigTest, TransferFromGroups) {
    milvus::ResourceGroupConfig config;
    config.AddTrnasferFromGroup("group_a");
    config.AddTrnasferFromGroup("group_b");
    EXPECT_EQ(config.TransferFromGroups().size(), 2);
    EXPECT_TRUE(config.TransferFromGroups().count("group_a"));
    EXPECT_TRUE(config.TransferFromGroups().count("group_b"));
}

TEST_F(ResourceGroupConfigTest, TransferToGroups) {
    milvus::ResourceGroupConfig config;
    config.AddTrnasferToGroup("group_x");
    config.AddTrnasferToGroup("group_y");
    EXPECT_EQ(config.TransferToGroups().size(), 2);
    EXPECT_TRUE(config.TransferToGroups().count("group_x"));
    EXPECT_TRUE(config.TransferToGroups().count("group_y"));
}

TEST_F(ResourceGroupConfigTest, NodeFilters) {
    milvus::ResourceGroupConfig config;
    config.AddNodeFilter("CPU", "32");
    config.AddNodeFilter("Memory", "64GB");
    EXPECT_EQ(config.NodeFilters().size(), 2);
    EXPECT_EQ(config.NodeFilters().at("CPU"), "32");
    EXPECT_EQ(config.NodeFilters().at("Memory"), "64GB");
}

class ResourceGroupDescTest : public ::testing::Test {};

TEST_F(ResourceGroupDescTest, DefaultValues) {
    milvus::ResourceGroupDesc desc;
    EXPECT_TRUE(desc.Name().empty());
    EXPECT_EQ(desc.Capacity(), 0u);
    EXPECT_EQ(desc.AvailableNodesNum(), 0u);
    EXPECT_TRUE(desc.LoadedReplicasNum().empty());
    EXPECT_TRUE(desc.OutgoingNodesNum().empty());
    EXPECT_TRUE(desc.IncomingNodesNum().empty());
    EXPECT_TRUE(desc.Nodes().empty());
}

TEST_F(ResourceGroupDescTest, NameGetterSetter) {
    milvus::ResourceGroupDesc desc;
    desc.SetName("rg_1");
    EXPECT_EQ(desc.Name(), "rg_1");
}

TEST_F(ResourceGroupDescTest, CapacityGetterSetter) {
    milvus::ResourceGroupDesc desc;
    desc.SetCapacity(100);
    EXPECT_EQ(desc.Capacity(), 100u);
}

TEST_F(ResourceGroupDescTest, AvailableNodesNumGetterSetter) {
    milvus::ResourceGroupDesc desc;
    desc.SetAvailableNodesNum(8);
    EXPECT_EQ(desc.AvailableNodesNum(), 8u);
}

TEST_F(ResourceGroupDescTest, LoadedReplicasNum) {
    milvus::ResourceGroupDesc desc;
    desc.AddLoadedReplicasNum("coll_a", 3);
    desc.AddLoadedReplicasNum("coll_b", 5);
    EXPECT_EQ(desc.LoadedReplicasNum().size(), 2);
    EXPECT_EQ(desc.LoadedReplicasNum().at("coll_a"), 3u);
    EXPECT_EQ(desc.LoadedReplicasNum().at("coll_b"), 5u);
}

TEST_F(ResourceGroupDescTest, OutgoingNodesNum) {
    milvus::ResourceGroupDesc desc;
    desc.AddOutgoingNodesNum("coll_x", 2);
    EXPECT_EQ(desc.OutgoingNodesNum().size(), 1);
    EXPECT_EQ(desc.OutgoingNodesNum().at("coll_x"), 2u);
}

TEST_F(ResourceGroupDescTest, IncomingNodesNum) {
    milvus::ResourceGroupDesc desc;
    desc.AddIncomingNodesNum("coll_y", 4);
    EXPECT_EQ(desc.IncomingNodesNum().size(), 1);
    EXPECT_EQ(desc.IncomingNodesNum().at("coll_y"), 4u);
}

TEST_F(ResourceGroupDescTest, SetConfig) {
    milvus::ResourceGroupDesc desc;
    milvus::ResourceGroupConfig config;
    config.SetRequests(3);
    config.SetLimits(10);
    desc.SetConfig(std::move(config));
    EXPECT_EQ(desc.Config().Requests(), 3u);
    EXPECT_EQ(desc.Config().Limits(), 10u);
}

TEST_F(ResourceGroupDescTest, AddNode) {
    milvus::ResourceGroupDesc desc;
    desc.AddNode(milvus::NodeInfo(1, "localhost:19530", "node-1"));
    desc.AddNode(milvus::NodeInfo(2, "localhost:19531", "node-2"));
    EXPECT_EQ(desc.Nodes().size(), 2);
    EXPECT_EQ(desc.Nodes()[0].id_, 1);
    EXPECT_EQ(desc.Nodes()[0].address_, "localhost:19530");
    EXPECT_EQ(desc.Nodes()[0].hostname_, "node-1");
    EXPECT_EQ(desc.Nodes()[1].id_, 2);
}
