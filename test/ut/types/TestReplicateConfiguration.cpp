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

class MilvusClusterTest : public ::testing::Test {};

TEST_F(MilvusClusterTest, GettersSettersAndFluentMethods) {
    milvus::MilvusCluster cluster;
    auto& ref = cluster.WithClusterID("cluster-a").WithUri("http://localhost:19530").WithToken("token");
    EXPECT_EQ(&ref, &cluster);
    EXPECT_EQ(cluster.ClusterID(), "cluster-a");
    EXPECT_EQ(cluster.Uri(), "http://localhost:19530");
    EXPECT_EQ(cluster.Token(), "token");

    cluster.SetPChannels({"by-dev-rootcoord-dml_0", "by-dev-rootcoord-dml_1"});
    cluster.AddPChannel("by-dev-rootcoord-dml_2");
    EXPECT_EQ(cluster.PChannels().size(), 3);
    EXPECT_EQ(cluster.PChannels()[2], "by-dev-rootcoord-dml_2");
}

class CrossClusterTopologyTest : public ::testing::Test {};

TEST_F(CrossClusterTopologyTest, GettersSettersAndFluentMethods) {
    milvus::CrossClusterTopology topology;
    auto& ref = topology.WithSourceClusterID("source").WithTargetClusterID("target");
    EXPECT_EQ(&ref, &topology);
    EXPECT_EQ(topology.SourceClusterID(), "source");
    EXPECT_EQ(topology.TargetClusterID(), "target");
}

class ReplicateConfigurationTest : public ::testing::Test {};

TEST_F(ReplicateConfigurationTest, GettersSettersAndFluentMethods) {
    milvus::ReplicateConfiguration configuration;

    milvus::MilvusCluster cluster;
    cluster.WithClusterID("cluster-a");
    auto& cluster_ref = configuration.AddCluster(std::move(cluster));
    EXPECT_EQ(&cluster_ref, &configuration);
    ASSERT_EQ(configuration.Clusters().size(), 1u);
    EXPECT_EQ(configuration.Clusters()[0].ClusterID(), "cluster-a");

    milvus::CrossClusterTopology topology;
    topology.WithSourceClusterID("source").WithTargetClusterID("target");
    auto& topology_ref = configuration.AddCrossClusterTopology(std::move(topology));
    EXPECT_EQ(&topology_ref, &configuration);
    ASSERT_EQ(configuration.CrossClusterTopologies().size(), 1u);
    EXPECT_EQ(configuration.CrossClusterTopologies()[0].SourceClusterID(), "source");

    std::vector<milvus::MilvusCluster> clusters;
    milvus::MilvusCluster new_cluster;
    new_cluster.WithClusterID("cluster-b");
    clusters.emplace_back(std::move(new_cluster));
    configuration.WithClusters(std::move(clusters));
    ASSERT_EQ(configuration.Clusters().size(), 1u);
    EXPECT_EQ(configuration.Clusters()[0].ClusterID(), "cluster-b");

    std::vector<milvus::CrossClusterTopology> topologies;
    milvus::CrossClusterTopology new_topology;
    new_topology.WithSourceClusterID("new-source").WithTargetClusterID("new-target");
    topologies.emplace_back(std::move(new_topology));
    configuration.WithCrossClusterTopologies(std::move(topologies));
    ASSERT_EQ(configuration.CrossClusterTopologies().size(), 1u);
    EXPECT_EQ(configuration.CrossClusterTopologies()[0].TargetClusterID(), "new-target");
}
