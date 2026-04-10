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

class CreateDatabaseRequestTest : public ::testing::Test {};

TEST_F(CreateDatabaseRequestTest, GettersAndSetters) {
    milvus::CreateDatabaseRequest req;

    req.WithDatabaseName("my_db");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    req.AddProperty("key1", "val1");
    EXPECT_EQ(req.Properties().at("key1"), "val1");

    req.AddProperty("key2", "val2");
    EXPECT_EQ(req.Properties().size(), 2);
}

TEST_F(CreateDatabaseRequestTest, SetProperties) {
    milvus::CreateDatabaseRequest req;
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

class DropDatabaseRequestTest : public ::testing::Test {};

TEST_F(DropDatabaseRequestTest, GettersAndSetters) {
    milvus::DropDatabaseRequest req;
    req.WithDatabaseName("drop_db");
    EXPECT_EQ(req.DatabaseName(), "drop_db");
}

class ListDatabasesRequestTest : public ::testing::Test {};

TEST_F(ListDatabasesRequestTest, DefaultConstruction) {
    milvus::ListDatabasesRequest req;
    // Just ensure it constructs without error
    (void)req;
}

class DescribeDatabaseRequestTest : public ::testing::Test {};

TEST_F(DescribeDatabaseRequestTest, GettersAndSetters) {
    milvus::DescribeDatabaseRequest req;
    req.WithDatabaseName("desc_db");
    EXPECT_EQ(req.DatabaseName(), "desc_db");
}

class AlterDatabasePropertiesRequestTest : public ::testing::Test {};

TEST_F(AlterDatabasePropertiesRequestTest, GettersAndSetters) {
    milvus::AlterDatabasePropertiesRequest req;

    req.WithDatabaseName("alter_db");
    EXPECT_EQ(req.DatabaseName(), "alter_db");

    req.AddProperty("k1", "v1");
    EXPECT_EQ(req.Properties().at("k1"), "v1");
}

TEST_F(AlterDatabasePropertiesRequestTest, SetProperties) {
    milvus::AlterDatabasePropertiesRequest req;
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

class DropDatabasePropertiesRequestTest : public ::testing::Test {};

TEST_F(DropDatabasePropertiesRequestTest, GettersAndSetters) {
    milvus::DropDatabasePropertiesRequest req;

    req.WithDatabaseName("drop_db_prop");
    EXPECT_EQ(req.DatabaseName(), "drop_db_prop");

    req.AddPropertyKey("k1");
    req.AddPropertyKey("k2");
    EXPECT_EQ(req.PropertyKeys().size(), 2);
    EXPECT_TRUE(req.PropertyKeys().count("k1"));
    EXPECT_TRUE(req.PropertyKeys().count("k2"));
}

TEST_F(DropDatabasePropertiesRequestTest, SetPropertyKeys) {
    milvus::DropDatabasePropertiesRequest req;
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
