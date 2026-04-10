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

class CheckHealthRequestTest : public ::testing::Test {};

TEST_F(CheckHealthRequestTest, DefaultConstruction) {
    milvus::CheckHealthRequest req;
    (void)req;
}

class FlushRequestTest : public ::testing::Test {};

TEST_F(FlushRequestTest, GettersAndSetters) {
    milvus::FlushRequest req;

    req.AddCollectionName("coll1");
    req.AddCollectionName("coll2");
    EXPECT_EQ(req.CollectionNames().size(), 2);
    EXPECT_TRUE(req.CollectionNames().count("coll1"));
    EXPECT_TRUE(req.CollectionNames().count("coll2"));

    req.WithWaitFlushedMs(30000);
    EXPECT_EQ(req.WaitFlushedMs(), 30000);
}

TEST_F(FlushRequestTest, AllMethods) {
    milvus::FlushRequest req;

    // DatabaseName
    EXPECT_TRUE(req.DatabaseName().empty());
    req.SetDatabaseName("db1");
    EXPECT_EQ(req.DatabaseName(), "db1");
    req.WithDatabaseName("db2");
    EXPECT_EQ(req.DatabaseName(), "db2");

    // SetCollectionNames / WithCollectionNames
    std::set<std::string> names{"c1", "c2"};
    req.SetCollectionNames(std::move(names));
    EXPECT_EQ(req.CollectionNames().size(), 2);
    EXPECT_TRUE(req.CollectionNames().count("c1"));

    std::set<std::string> names2{"x1"};
    req.WithCollectionNames(std::move(names2));
    EXPECT_EQ(req.CollectionNames().size(), 1);
    EXPECT_TRUE(req.CollectionNames().count("x1"));

    // SetWaitFlushedMs
    req.SetWaitFlushedMs(5000);
    EXPECT_EQ(req.WaitFlushedMs(), 5000);
}

class CompactRequestTest : public ::testing::Test {};

TEST_F(CompactRequestTest, GettersAndSetters) {
    milvus::CompactRequest req;
    req.WithCollectionName("compact_coll");
    EXPECT_EQ(req.CollectionName(), "compact_coll");
}

TEST_F(CompactRequestTest, AllMethods) {
    milvus::CompactRequest req;

    // DatabaseName
    EXPECT_TRUE(req.DatabaseName().empty());
    req.SetDatabaseName("db1");
    EXPECT_EQ(req.DatabaseName(), "db1");
    req.WithDatabaseName("db2");
    EXPECT_EQ(req.DatabaseName(), "db2");

    // SetCollectionName
    req.SetCollectionName("coll1");
    EXPECT_EQ(req.CollectionName(), "coll1");

    // SetClusteringCompaction
    req.SetClusteringCompaction(true);
    EXPECT_TRUE(req.ClusteringCompaction());
    req.SetClusteringCompaction(false);
    EXPECT_FALSE(req.ClusteringCompaction());
}

class GetCompactionStateRequestTest : public ::testing::Test {};

TEST_F(GetCompactionStateRequestTest, GettersAndSetters) {
    milvus::GetCompactionStateRequest req;
    req.WithCompactionID(12345);
    EXPECT_EQ(req.CompactionID(), 12345);
}

class GetCompactionPlansRequestTest : public ::testing::Test {};

TEST_F(GetCompactionPlansRequestTest, GettersAndSetters) {
    milvus::GetCompactionPlansRequest req;
    req.WithCompactionID(67890);
    EXPECT_EQ(req.CompactionID(), 67890);
}

TEST_F(GetCompactionStateRequestTest, SetMethod) {
    milvus::GetCompactionStateRequest req;
    req.SetCompactionID(99999);
    EXPECT_EQ(req.CompactionID(), 99999);
}

TEST_F(GetCompactionPlansRequestTest, SetMethod) {
    milvus::GetCompactionPlansRequest req;
    req.SetCompactionID(88888);
    EXPECT_EQ(req.CompactionID(), 88888);
}

class ListPersistentSegmentsRequestTest : public ::testing::Test {};

TEST_F(ListPersistentSegmentsRequestTest, GettersAndSetters) {
    milvus::ListPersistentSegmentsRequest req;

    req.WithCollectionName("seg_coll");
    EXPECT_EQ(req.CollectionName(), "seg_coll");

    req.WithDatabaseName("seg_db");
    EXPECT_EQ(req.DatabaseName(), "seg_db");
}

class ListQuerySegmentsRequestTest : public ::testing::Test {};

TEST_F(ListQuerySegmentsRequestTest, GettersAndSetters) {
    milvus::ListQuerySegmentsRequest req;

    req.WithCollectionName("qseg_coll");
    EXPECT_EQ(req.CollectionName(), "qseg_coll");

    req.WithDatabaseName("qseg_db");
    EXPECT_EQ(req.DatabaseName(), "qseg_db");
}

class RunAnalyzerRequestTest : public ::testing::Test {};

TEST_F(RunAnalyzerRequestTest, GettersAndSetters) {
    milvus::RunAnalyzerRequest req;

    req.AddText("hello world");
    req.AddText("foo bar");
    EXPECT_EQ(req.Texts().size(), 2);
    EXPECT_EQ(req.Texts()[0], "hello world");

    std::vector<std::string> texts{"a", "b", "c"};
    req.WithTexts(texts);
    EXPECT_EQ(req.Texts().size(), 3);

    nlohmann::json params = {{"type", "standard"}};
    req.WithAnalyzerParams(params);
    EXPECT_EQ(req.AnalyzerParams()["type"], "standard");

    req.WithDetail(true);
    EXPECT_TRUE(req.IsWithDetail());

    req.WithHash(true);
    EXPECT_TRUE(req.IsWithHash());
}

class TransferNodeRequestTest : public ::testing::Test {};

TEST_F(TransferNodeRequestTest, GettersAndWithSetters) {
    milvus::TransferNodeRequest req;

    auto& ref1 = req.WithSourceGroup("src_group");
    EXPECT_EQ(req.SourceGroup(), "src_group");
    EXPECT_EQ(&ref1, &req);

    auto& ref2 = req.WithTargetGroup("tgt_group");
    EXPECT_EQ(req.TargetGroup(), "tgt_group");
    EXPECT_EQ(&ref2, &req);

    auto& ref3 = req.WithNumNodes(5);
    EXPECT_EQ(req.NumNodes(), 5);
    EXPECT_EQ(&ref3, &req);
}

TEST_F(TransferNodeRequestTest, FluentChaining) {
    milvus::TransferNodeRequest req;
    auto& ref = req.WithSourceGroup("s").WithTargetGroup("t").WithNumNodes(3);
    EXPECT_EQ(&ref, &req);
    EXPECT_EQ(req.SourceGroup(), "s");
    EXPECT_EQ(req.TargetGroup(), "t");
    EXPECT_EQ(req.NumNodes(), 3);
}

class TransferReplicaRequestTest : public ::testing::Test {};

TEST_F(TransferReplicaRequestTest, GettersAndWithSetters) {
    milvus::TransferReplicaRequest req;

    auto& ref1 = req.WithSourceGroup("src_rg");
    EXPECT_EQ(req.SourceGroup(), "src_rg");
    EXPECT_EQ(&ref1, &req);

    auto& ref2 = req.WithTargetGroup("tgt_rg");
    EXPECT_EQ(req.TargetGroup(), "tgt_rg");
    EXPECT_EQ(&ref2, &req);

    auto& ref3 = req.WithNumReplicas(4);
    EXPECT_EQ(req.NumReplicas(), 4);
    EXPECT_EQ(&ref3, &req);

    auto& ref4 = req.WithDatabaseName("mydb");
    EXPECT_EQ(req.DatabaseName(), "mydb");
    EXPECT_EQ(&ref4, &req);

    auto& ref5 = req.WithCollectionName("mycoll");
    EXPECT_EQ(req.CollectionName(), "mycoll");
    EXPECT_EQ(&ref5, &req);
}

TEST_F(TransferReplicaRequestTest, FluentChaining) {
    milvus::TransferReplicaRequest req;
    auto& ref =
        req.WithSourceGroup("s").WithTargetGroup("t").WithNumReplicas(2).WithCollectionName("coll").WithDatabaseName(
            "db");
    EXPECT_EQ(&ref, &req);
    EXPECT_EQ(req.SourceGroup(), "s");
    EXPECT_EQ(req.TargetGroup(), "t");
    EXPECT_EQ(req.NumReplicas(), 2);
}

class UpdateResourceGroupsRequestTest : public ::testing::Test {};

TEST_F(UpdateResourceGroupsRequestTest, AddGroup) {
    milvus::UpdateResourceGroupsRequest req;

    milvus::ResourceGroupConfig config1;
    config1.SetRequests(3);
    config1.SetLimits(5);
    auto& ref1 = req.AddGroup("rg1", std::move(config1));
    EXPECT_EQ(&ref1, &req);

    milvus::ResourceGroupConfig config2;
    config2.SetRequests(1);
    auto& ref2 = req.AddGroup("rg2", std::move(config2));
    EXPECT_EQ(&ref2, &req);

    EXPECT_EQ(req.Groups().size(), 2);
    EXPECT_EQ(req.Groups().at("rg1").Requests(), 3);
    EXPECT_EQ(req.Groups().at("rg1").Limits(), 5);
    EXPECT_EQ(req.Groups().at("rg2").Requests(), 1);
}

TEST_F(UpdateResourceGroupsRequestTest, SetAndWithGroups) {
    milvus::UpdateResourceGroupsRequest req;

    // SetGroups
    std::unordered_map<std::string, milvus::ResourceGroupConfig> groups;
    milvus::ResourceGroupConfig cfg;
    cfg.SetRequests(10);
    groups["rg_a"] = std::move(cfg);
    req.SetGroups(std::move(groups));
    EXPECT_EQ(req.Groups().size(), 1);
    EXPECT_EQ(req.Groups().at("rg_a").Requests(), 10);

    // WithGroups (resets)
    std::unordered_map<std::string, milvus::ResourceGroupConfig> groups2;
    milvus::ResourceGroupConfig cfg2;
    cfg2.SetRequests(20);
    groups2["rg_b"] = std::move(cfg2);
    milvus::ResourceGroupConfig cfg3;
    cfg3.SetRequests(30);
    groups2["rg_c"] = std::move(cfg3);
    req.WithGroups(std::move(groups2));
    EXPECT_EQ(req.Groups().size(), 2);
    EXPECT_EQ(req.Groups().at("rg_b").Requests(), 20);
    EXPECT_EQ(req.Groups().at("rg_c").Requests(), 30);
}

class ResourceGroupRequestTest : public ::testing::Test {};

TEST_F(ResourceGroupRequestTest, GettersAndSetters) {
    milvus::ResourceGroupRequest req;

    EXPECT_TRUE(req.GroupName().empty());

    req.SetGroupName("rg1");
    EXPECT_EQ(req.GroupName(), "rg1");

    auto& ref = req.WithGroupName("rg2");
    EXPECT_EQ(req.GroupName(), "rg2");
    EXPECT_EQ(&ref, &req);
}

TEST_F(ResourceGroupRequestTest, Aliases) {
    // DropResourceGroupRequest is an alias for ResourceGroupRequest
    milvus::DropResourceGroupRequest drop_req;
    drop_req.WithGroupName("drop_rg");
    EXPECT_EQ(drop_req.GroupName(), "drop_rg");

    // DescribeResourceGroupRequest is an alias for ResourceGroupRequest
    milvus::DescribeResourceGroupRequest desc_req;
    desc_req.WithGroupName("desc_rg");
    EXPECT_EQ(desc_req.GroupName(), "desc_rg");
}

class CreateResourceGroupRequestTest : public ::testing::Test {};

TEST_F(CreateResourceGroupRequestTest, GettersAndSetters) {
    milvus::CreateResourceGroupRequest req;

    EXPECT_TRUE(req.Name().empty());

    req.SetName("new_rg");
    EXPECT_EQ(req.Name(), "new_rg");

    auto& ref = req.WithName("rg2");
    EXPECT_EQ(req.Name(), "rg2");
    EXPECT_EQ(&ref, &req);
}

TEST_F(CreateResourceGroupRequestTest, ConfigSetterAndGetter) {
    milvus::CreateResourceGroupRequest req;

    // SetConfig
    milvus::ResourceGroupConfig config;
    config.SetRequests(5);
    config.SetLimits(10);
    req.SetConfig(std::move(config));
    EXPECT_EQ(req.Config().Requests(), 5);
    EXPECT_EQ(req.Config().Limits(), 10);

    // WithConfig
    milvus::ResourceGroupConfig config2;
    config2.SetRequests(8);
    auto& ref = req.WithConfig(std::move(config2));
    EXPECT_EQ(req.Config().Requests(), 8);
    EXPECT_EQ(&ref, &req);
}

class TransferReplicaRequestSetTest : public ::testing::Test {};

TEST_F(TransferReplicaRequestSetTest, SetMethods) {
    milvus::TransferReplicaRequest req;

    req.SetDatabaseName("db1");
    EXPECT_EQ(req.DatabaseName(), "db1");

    req.SetCollectionName("coll1");
    EXPECT_EQ(req.CollectionName(), "coll1");

    req.SetSourceGroup("src");
    EXPECT_EQ(req.SourceGroup(), "src");

    req.SetTargetGroup("tgt");
    EXPECT_EQ(req.TargetGroup(), "tgt");

    req.SetNumReplicas(7);
    EXPECT_EQ(req.NumReplicas(), 7);
}

class TransferNodeRequestSetTest : public ::testing::Test {};

TEST_F(TransferNodeRequestSetTest, SetMethods) {
    milvus::TransferNodeRequest req;

    req.SetSourceGroup("src");
    EXPECT_EQ(req.SourceGroup(), "src");

    req.SetTargetGroup("tgt");
    EXPECT_EQ(req.TargetGroup(), "tgt");

    req.SetNumNodes(3);
    EXPECT_EQ(req.NumNodes(), 3);
}

TEST_F(CompactRequestTest, ClusteringCompaction) {
    milvus::CompactRequest req;

    // Default is false
    EXPECT_FALSE(req.ClusteringCompaction());

    auto& ref = req.WithClusteringCompaction(true);
    EXPECT_TRUE(req.ClusteringCompaction());
    EXPECT_EQ(&ref, &req);

    req.WithClusteringCompaction(false);
    EXPECT_FALSE(req.ClusteringCompaction());
}
