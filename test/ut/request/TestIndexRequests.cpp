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

class CreateIndexRequestTest : public ::testing::Test {};

TEST_F(CreateIndexRequestTest, GettersAndSetters) {
    milvus::CreateIndexRequest req;

    req.WithCollectionName("idx_coll");
    EXPECT_EQ(req.CollectionName(), "idx_coll");

    milvus::IndexDesc idx;
    req.AddIndex(std::move(idx));
    EXPECT_EQ(req.Indexes().size(), 1);

    req.WithSync(false);
    EXPECT_FALSE(req.Sync());

    req.WithTimeoutMs(90000);
    EXPECT_EQ(req.TimeoutMs(), 90000);
}

TEST_F(CreateIndexRequestTest, IndexRequestBaseMethods) {
    milvus::CreateIndexRequest req;

    EXPECT_TRUE(req.DatabaseName().empty());
    req.SetDatabaseName("test_db");
    EXPECT_EQ(req.DatabaseName(), "test_db");

    milvus::CreateIndexRequest req2;
    req2.WithDatabaseName("db2");
    EXPECT_EQ(req2.DatabaseName(), "db2");
}

class DescribeIndexRequestTest : public ::testing::Test {};

TEST_F(DescribeIndexRequestTest, GettersAndSetters) {
    milvus::DescribeIndexRequest req;

    req.WithCollectionName("desc_idx_coll");
    EXPECT_EQ(req.CollectionName(), "desc_idx_coll");

    req.WithFieldName("vec_field");
    EXPECT_EQ(req.FieldName(), "vec_field");

    req.WithIndexName("my_index");
    EXPECT_EQ(req.IndexName(), "my_index");
}

class DropIndexRequestTest : public ::testing::Test {};

TEST_F(DropIndexRequestTest, GettersAndSetters) {
    milvus::DropIndexRequest req;

    req.WithCollectionName("drop_idx_coll");
    EXPECT_EQ(req.CollectionName(), "drop_idx_coll");

    req.WithFieldName("vec_field");
    EXPECT_EQ(req.FieldName(), "vec_field");

    req.WithIndexName("my_index");
    EXPECT_EQ(req.IndexName(), "my_index");
}

class ListIndexesRequestTest : public ::testing::Test {};

TEST_F(ListIndexesRequestTest, GettersAndSetters) {
    milvus::ListIndexesRequest req;
    req.WithCollectionName("list_idx_coll");
    EXPECT_EQ(req.CollectionName(), "list_idx_coll");
}

class AlterIndexPropertiesRequestTest : public ::testing::Test {};

TEST_F(AlterIndexPropertiesRequestTest, GettersAndSetters) {
    milvus::AlterIndexPropertiesRequest req;

    req.WithCollectionName("alter_idx_coll");
    EXPECT_EQ(req.CollectionName(), "alter_idx_coll");

    req.WithIndexName("my_index");
    EXPECT_EQ(req.IndexName(), "my_index");

    req.AddProperty("mmap.enabled", "true");
    EXPECT_EQ(req.Properties().at("mmap.enabled"), "true");
}

class DropIndexPropertiesRequestTest : public ::testing::Test {};

TEST_F(DropIndexPropertiesRequestTest, GettersAndSetters) {
    milvus::DropIndexPropertiesRequest req;

    req.WithCollectionName("drop_idx_prop_coll");
    EXPECT_EQ(req.CollectionName(), "drop_idx_prop_coll");

    req.WithIndexName("my_index");
    EXPECT_EQ(req.IndexName(), "my_index");

    req.AddPropertyKey("mmap.enabled");
    EXPECT_EQ(req.PropertyKeys().size(), 1);
    EXPECT_TRUE(req.PropertyKeys().count("mmap.enabled"));
}

TEST_F(DescribeIndexRequestTest, SetIndexNameAndTimestamp) {
    milvus::DescribeIndexRequest req;
    req.SetIndexName("idx_set");
    EXPECT_EQ(req.IndexName(), "idx_set");

    req.SetFieldName("field_set");
    EXPECT_EQ(req.FieldName(), "field_set");

    req.SetTimestamp(12345);
    EXPECT_EQ(req.Timestamp(), 12345);

    auto& ref = req.WithTimestamp(67890);
    EXPECT_EQ(req.Timestamp(), 67890);
    EXPECT_EQ(&ref, &req);
}

TEST_F(AlterIndexPropertiesRequestTest, SetIndexNameAndProperties) {
    milvus::AlterIndexPropertiesRequest req;
    req.SetIndexName("idx_alter");
    EXPECT_EQ(req.IndexName(), "idx_alter");

    std::unordered_map<std::string, std::string> props = {{"k1", "v1"}};
    req.SetProperties(std::move(props));
    EXPECT_EQ(req.Properties().size(), 1);
    EXPECT_EQ(req.Properties().at("k1"), "v1");

    std::unordered_map<std::string, std::string> props2 = {{"k2", "v2"}};
    auto& ref = req.WithProperties(std::move(props2));
    EXPECT_EQ(req.Properties().size(), 1);
    EXPECT_EQ(req.Properties().at("k2"), "v2");
    EXPECT_EQ(&ref, &req);
}

TEST_F(DropIndexPropertiesRequestTest, SetPropertyKeysAndIndexName) {
    milvus::DropIndexPropertiesRequest req;
    req.SetIndexName("idx_drop");
    EXPECT_EQ(req.IndexName(), "idx_drop");

    std::set<std::string> keys = {"k1", "k2"};
    req.SetPropertyKeys(std::move(keys));
    EXPECT_EQ(req.PropertyKeys().size(), 2);
    EXPECT_TRUE(req.PropertyKeys().count("k1"));

    std::set<std::string> keys2 = {"k3"};
    auto& ref = req.WithPropertyKeys(std::move(keys2));
    EXPECT_EQ(req.PropertyKeys().size(), 1);
    EXPECT_TRUE(req.PropertyKeys().count("k3"));
    EXPECT_EQ(&ref, &req);
}
