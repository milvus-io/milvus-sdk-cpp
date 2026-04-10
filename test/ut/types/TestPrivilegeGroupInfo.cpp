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

class PrivilegeGroupInfoTest : public ::testing::Test {};

TEST_F(PrivilegeGroupInfoTest, DefaultConstructor) {
    milvus::PrivilegeGroupInfo info;
    EXPECT_TRUE(info.Name().empty());
    EXPECT_TRUE(info.Privileges().empty());
}

TEST_F(PrivilegeGroupInfoTest, ParameterizedConstructor) {
    std::vector<std::string> privs = {"Insert", "Query", "Delete"};
    milvus::PrivilegeGroupInfo info("admin_group", std::move(privs));
    EXPECT_EQ(info.Name(), "admin_group");
    EXPECT_EQ(info.Privileges().size(), 3);
    EXPECT_EQ(info.Privileges()[0], "Insert");
    EXPECT_EQ(info.Privileges()[1], "Query");
    EXPECT_EQ(info.Privileges()[2], "Delete");
}

TEST_F(PrivilegeGroupInfoTest, SetName) {
    milvus::PrivilegeGroupInfo info;
    info.SetName("my_group");
    EXPECT_EQ(info.Name(), "my_group");
}

TEST_F(PrivilegeGroupInfoTest, AddPrivilege) {
    milvus::PrivilegeGroupInfo info;
    info.AddPrivilege("Search");
    info.AddPrivilege("CreateCollection");
    EXPECT_EQ(info.Privileges().size(), 2);
    EXPECT_EQ(info.Privileges()[0], "Search");
    EXPECT_EQ(info.Privileges()[1], "CreateCollection");
}
