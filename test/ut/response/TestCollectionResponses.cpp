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

class HasCollectionResponseTest : public ::testing::Test {};

TEST_F(HasCollectionResponseTest, SetterAndGetter) {
    milvus::HasCollectionResponse resp;
    EXPECT_FALSE(resp.Has());

    resp.SetHas(true);
    EXPECT_TRUE(resp.Has());

    resp.SetHas(false);
    EXPECT_FALSE(resp.Has());
}

class ListCollectionsResponseTest : public ::testing::Test {};

TEST_F(ListCollectionsResponseTest, SetterAndGetter) {
    milvus::ListCollectionsResponse resp;

    std::vector<std::string> names{"coll1", "coll2"};
    resp.SetCollectionNames(std::move(names));
    EXPECT_EQ(resp.CollectionNames().size(), 2);
    EXPECT_EQ(resp.CollectionNames()[0], "coll1");

    std::vector<milvus::CollectionInfo> infos;
    milvus::CollectionInfo info;
    infos.push_back(info);
    resp.SetCollectionInfos(std::move(infos));
    EXPECT_EQ(resp.CollectionInfos().size(), 1);
}

class DescribeCollectionResponseTest : public ::testing::Test {};

TEST_F(DescribeCollectionResponseTest, SetterAndGetter) {
    milvus::DescribeCollectionResponse resp;
    milvus::CollectionDesc desc;
    resp.SetDesc(std::move(desc));
    (void)resp.Desc();
}

class BatchDescribeCollectionsResponseTest : public ::testing::Test {};

TEST_F(BatchDescribeCollectionsResponseTest, SetterAndGetter) {
    milvus::BatchDescribeCollectionsResponse resp;

    milvus::CollectionSchema schema1("coll1");
    milvus::CollectionDesc desc1;
    desc1.SetID(101);
    desc1.SetSchema(std::move(schema1));

    milvus::CollectionSchema schema2("coll2");
    milvus::CollectionDesc desc2;
    desc2.SetID(102);
    desc2.SetSchema(std::move(schema2));

    std::vector<milvus::CollectionDesc> descs;
    descs.emplace_back(std::move(desc1));
    descs.emplace_back(std::move(desc2));
    resp.SetDescs(std::move(descs));

    ASSERT_EQ(resp.Descs().size(), 2);
    EXPECT_EQ(resp.Descs()[0].ID(), 101);
    EXPECT_EQ(resp.Descs()[0].CollectionName(), "coll1");
    EXPECT_EQ(resp.Descs()[1].ID(), 102);
    EXPECT_EQ(resp.Descs()[1].CollectionName(), "coll2");
}

class DescribeReplicasResponseTest : public ::testing::Test {};

TEST_F(DescribeReplicasResponseTest, SetterAndGetter) {
    milvus::ShardReplica shard;
    shard.SetLeaderID(11);
    shard.SetLeaderAddress("127.0.0.1:19530");
    shard.SetChannelName("by-dev-rootcoord-dml_0");
    shard.SetNodeIDs({1, 2});

    milvus::ReplicaInfo replica;
    replica.SetReplicaID(1001);
    replica.SetCollectionID(2002);
    replica.SetPartitionIDs({3003, 3004});
    replica.SetShardReplicas({std::move(shard)});
    replica.SetNodeIDs({1, 2, 3});
    replica.SetResourceGroupName("rg1");
    replica.SetNumOutboundNode({{"rg2", 1}});

    milvus::DescribeReplicasResponse resp;
    resp.SetReplicas({std::move(replica)});

    ASSERT_EQ(resp.Replicas().size(), 1);
    const auto& actual = resp.Replicas()[0];
    EXPECT_EQ(actual.ReplicaID(), 1001);
    EXPECT_EQ(actual.CollectionID(), 2002);
    ASSERT_EQ(actual.PartitionIDs().size(), 2);
    EXPECT_EQ(actual.PartitionIDs()[0], 3003);
    ASSERT_EQ(actual.ShardReplicas().size(), 1);
    EXPECT_EQ(actual.ShardReplicas()[0].LeaderID(), 11);
    EXPECT_EQ(actual.ShardReplicas()[0].LeaderAddress(), "127.0.0.1:19530");
    EXPECT_EQ(actual.ShardReplicas()[0].ChannelName(), "by-dev-rootcoord-dml_0");
    ASSERT_EQ(actual.ShardReplicas()[0].NodeIDs().size(), 2);
    EXPECT_EQ(actual.ShardReplicas()[0].NodeIDs()[1], 2);
    ASSERT_EQ(actual.NodeIDs().size(), 3);
    EXPECT_EQ(actual.NodeIDs()[2], 3);
    EXPECT_EQ(actual.ResourceGroupName(), "rg1");
    EXPECT_EQ(actual.NumOutboundNode().at("rg2"), 1);
}

class GetCollectionStatsResponseTest : public ::testing::Test {};

TEST_F(GetCollectionStatsResponseTest, SetterAndGetter) {
    milvus::GetCollectionStatsResponse resp;
    milvus::CollectionStat stats;
    resp.SetStats(std::move(stats));
    (void)resp.Stats();
}

class GetLoadStateResponseTest : public ::testing::Test {};

TEST_F(GetLoadStateResponseTest, SetterAndGetter) {
    milvus::GetLoadStateResponse resp;

    resp.SetState(milvus::LoadState::LOAD_STATE_LOADED);
    EXPECT_EQ(resp.State(), milvus::LoadState::LOAD_STATE_LOADED);

    resp.SetProgress(100);
    EXPECT_EQ(resp.Progress(), 100);
}
