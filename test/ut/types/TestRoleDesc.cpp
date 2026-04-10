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

class GrantItemTest : public ::testing::Test {};

TEST_F(GrantItemTest, Constructor) {
    milvus::GrantItem item("Global", "*", "default", "admin", "root", "Insert");
    EXPECT_EQ(item.object_type_, "Global");
    EXPECT_EQ(item.object_name_, "*");
    EXPECT_EQ(item.db_name_, "default");
    EXPECT_EQ(item.role_name_, "admin");
    EXPECT_EQ(item.grantor_name_, "root");
    EXPECT_EQ(item.privilege_, "Insert");
}

class RoleDescTest : public ::testing::Test {};

TEST_F(RoleDescTest, DefaultConstructor) {
    milvus::RoleDesc desc;
    EXPECT_TRUE(desc.Name().empty());
    EXPECT_TRUE(desc.GrantItems().empty());
}

TEST_F(RoleDescTest, ParameterizedConstructor) {
    std::vector<milvus::GrantItem> items;
    items.emplace_back("Global", "*", "db1", "role1", "root", "Query");
    milvus::RoleDesc desc("my_role", std::move(items));
    EXPECT_EQ(desc.Name(), "my_role");
    EXPECT_EQ(desc.GrantItems().size(), 1);
    EXPECT_EQ(desc.GrantItems()[0].privilege_, "Query");
}

TEST_F(RoleDescTest, SetName) {
    milvus::RoleDesc desc;
    desc.SetName("admin");
    EXPECT_EQ(desc.Name(), "admin");
}

TEST_F(RoleDescTest, AddGrantItem) {
    milvus::RoleDesc desc;
    desc.SetName("editor");
    desc.AddGrantItem(milvus::GrantItem("Collection", "coll_1", "db1", "editor", "root", "Insert"));
    desc.AddGrantItem(milvus::GrantItem("Collection", "coll_1", "db1", "editor", "root", "Delete"));
    EXPECT_EQ(desc.GrantItems().size(), 2);
    EXPECT_EQ(desc.GrantItems()[0].privilege_, "Insert");
    EXPECT_EQ(desc.GrantItems()[1].privilege_, "Delete");
}
