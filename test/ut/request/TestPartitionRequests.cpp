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

class CreatePartitionRequestTest : public ::testing::Test {};

TEST_F(CreatePartitionRequestTest, GettersAndSetters) {
    milvus::CreatePartitionRequest req;
    req.WithCollectionName("test_coll");
    EXPECT_EQ(req.CollectionName(), "test_coll");

    req.WithPartitionName("test_part");
    EXPECT_EQ(req.PartitionName(), "test_part");
}

TEST_F(CreatePartitionRequestTest, PartitionRequestBaseMethods) {
    milvus::CreatePartitionRequest req;

    EXPECT_TRUE(req.DatabaseName().empty());
    req.SetDatabaseName("test_db");
    EXPECT_EQ(req.DatabaseName(), "test_db");

    milvus::CreatePartitionRequest req2;
    req2.WithDatabaseName("db2");
    EXPECT_EQ(req2.DatabaseName(), "db2");
}

class DropPartitionRequestTest : public ::testing::Test {};

TEST_F(DropPartitionRequestTest, GettersAndSetters) {
    milvus::DropPartitionRequest req;
    req.WithCollectionName("test_coll");
    EXPECT_EQ(req.CollectionName(), "test_coll");

    req.WithPartitionName("test_part");
    EXPECT_EQ(req.PartitionName(), "test_part");
}

class HasPartitionRequestTest : public ::testing::Test {};

TEST_F(HasPartitionRequestTest, GettersAndSetters) {
    milvus::HasPartitionRequest req;
    req.WithCollectionName("test_coll");
    EXPECT_EQ(req.CollectionName(), "test_coll");

    req.WithPartitionName("test_part");
    EXPECT_EQ(req.PartitionName(), "test_part");
}

class ListPartitionsRequestTest : public ::testing::Test {};

TEST_F(ListPartitionsRequestTest, GettersAndSetters) {
    milvus::ListPartitionsRequest req;
    req.WithCollectionName("test_coll");
    EXPECT_EQ(req.CollectionName(), "test_coll");

    req.WithDatabaseName("test_db");
    EXPECT_EQ(req.DatabaseName(), "test_db");
}

class LoadPartitionsRequestTest : public ::testing::Test {};

TEST_F(LoadPartitionsRequestTest, GettersAndSetters) {
    milvus::LoadPartitionsRequest req;

    req.WithCollectionName("test_coll");
    EXPECT_EQ(req.CollectionName(), "test_coll");

    req.AddPartitionName("p1");
    req.AddPartitionName("p2");
    EXPECT_EQ(req.PartitionNames().size(), 2);
    EXPECT_TRUE(req.PartitionNames().count("p1"));

    req.WithSync(false);
    EXPECT_FALSE(req.Sync());

    req.WithReplicaNum(5);
    EXPECT_EQ(req.ReplicaNum(), 5);

    req.WithTimeoutMs(120000);
    EXPECT_EQ(req.TimeoutMs(), 120000);
}

class ReleasePartitionsRequestTest : public ::testing::Test {};

TEST_F(ReleasePartitionsRequestTest, GettersAndSetters) {
    milvus::ReleasePartitionsRequest req;

    req.WithCollectionName("test_coll");
    EXPECT_EQ(req.CollectionName(), "test_coll");

    req.AddPartitionName("p1");
    req.AddPartitionName("p2");
    EXPECT_EQ(req.PartitionNames().size(), 2);
    EXPECT_TRUE(req.PartitionNames().count("p1"));
    EXPECT_TRUE(req.PartitionNames().count("p2"));
}

TEST_F(LoadPartitionsRequestTest, SetDatabaseName) {
    milvus::LoadPartitionsRequest req;
    req.SetDatabaseName("my_db");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    auto& ref = req.WithDatabaseName("other_db");
    EXPECT_EQ(req.DatabaseName(), "other_db");
    EXPECT_EQ(&ref, &req);
}

TEST_F(LoadPartitionsRequestTest, SetPartitionNames) {
    milvus::LoadPartitionsRequest req;
    std::set<std::string> names = {"p1", "p2"};
    req.SetPartitionNames(names);
    EXPECT_EQ(req.PartitionNames().size(), 2);
    EXPECT_TRUE(req.PartitionNames().count("p1"));

    std::set<std::string> names2 = {"p3"};
    auto& ref = req.WithPartitionNames(names2);
    EXPECT_EQ(req.PartitionNames().size(), 1);
    EXPECT_TRUE(req.PartitionNames().count("p3"));
    EXPECT_EQ(&ref, &req);
}

TEST_F(LoadPartitionsRequestTest, SetRefresh) {
    milvus::LoadPartitionsRequest req;
    EXPECT_FALSE(req.Refresh());
    req.SetRefresh(true);
    EXPECT_TRUE(req.Refresh());

    auto& ref = req.WithRefresh(false);
    EXPECT_FALSE(req.Refresh());
    EXPECT_EQ(&ref, &req);
}

TEST_F(LoadPartitionsRequestTest, SetLoadFields) {
    milvus::LoadPartitionsRequest req;
    EXPECT_TRUE(req.LoadFields().empty());

    std::set<std::string> fields = {"f1", "f2"};
    req.SetLoadFields(fields);
    EXPECT_EQ(req.LoadFields().size(), 2);

    std::set<std::string> fields2 = {"f3"};
    auto& ref = req.WithLoadFields(fields2);
    EXPECT_EQ(req.LoadFields().size(), 1);
    EXPECT_TRUE(req.LoadFields().count("f3"));
    EXPECT_EQ(&ref, &req);

    req.AddLoadField("f4");
    EXPECT_EQ(req.LoadFields().size(), 2);
    EXPECT_TRUE(req.LoadFields().count("f4"));
}

TEST_F(LoadPartitionsRequestTest, SetSkipDynamicField) {
    milvus::LoadPartitionsRequest req;
    EXPECT_FALSE(req.SkipDynamicField());
    req.SetSkipDynamicField(true);
    EXPECT_TRUE(req.SkipDynamicField());

    auto& ref = req.WithSkipDynamicField(false);
    EXPECT_FALSE(req.SkipDynamicField());
    EXPECT_EQ(&ref, &req);
}

TEST_F(LoadPartitionsRequestTest, SetTargetResourceGroups) {
    milvus::LoadPartitionsRequest req;
    EXPECT_TRUE(req.TargetResourceGroups().empty());

    std::set<std::string> groups = {"rg1", "rg2"};
    req.SetTargetResourceGroups(groups);
    EXPECT_EQ(req.TargetResourceGroups().size(), 2);

    std::set<std::string> groups2 = {"rg3"};
    auto& ref = req.WithTargetResourceGroups(groups2);
    EXPECT_EQ(req.TargetResourceGroups().size(), 1);
    EXPECT_TRUE(req.TargetResourceGroups().count("rg3"));
    EXPECT_EQ(&ref, &req);

    req.AddTargetResourceGroups("rg4");
    EXPECT_EQ(req.TargetResourceGroups().size(), 2);
    EXPECT_TRUE(req.TargetResourceGroups().count("rg4"));
}

TEST_F(ReleasePartitionsRequestTest, SetDatabaseName) {
    milvus::ReleasePartitionsRequest req;
    req.SetDatabaseName("my_db");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    auto& ref = req.WithDatabaseName("other_db");
    EXPECT_EQ(req.DatabaseName(), "other_db");
    EXPECT_EQ(&ref, &req);
}

TEST_F(ReleasePartitionsRequestTest, SetPartitionNames) {
    milvus::ReleasePartitionsRequest req;
    std::set<std::string> names = {"p1", "p2"};
    req.SetPartitionNames(names);
    EXPECT_EQ(req.PartitionNames().size(), 2);
    EXPECT_TRUE(req.PartitionNames().count("p1"));

    std::set<std::string> names2 = {"p3"};
    auto& ref = req.WithPartitionNames(names2);
    EXPECT_EQ(req.PartitionNames().size(), 1);
    EXPECT_TRUE(req.PartitionNames().count("p3"));
    EXPECT_EQ(&ref, &req);
}

class GetPartitionStatsRequestTest : public ::testing::Test {};

TEST_F(GetPartitionStatsRequestTest, GettersAndSetters) {
    milvus::GetPartitionStatsRequest req;
    req.WithCollectionName("test_coll");
    EXPECT_EQ(req.CollectionName(), "test_coll");

    req.WithPartitionName("test_part");
    EXPECT_EQ(req.PartitionName(), "test_part");
}
