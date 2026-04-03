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

class ListUsersResponseTest : public ::testing::Test {};

TEST_F(ListUsersResponseTest, SetterAndGetter) {
    milvus::ListUsersResponse resp;
    std::vector<std::string> users{"user1", "user2"};
    resp.SetUserNames(std::move(users));
    EXPECT_EQ(resp.UserNames().size(), 2);
}

class ListRolesResponseTest : public ::testing::Test {};

TEST_F(ListRolesResponseTest, SetterAndGetter) {
    milvus::ListRolesResponse resp;
    std::vector<std::string> roles{"role1", "role2"};
    resp.SetRoleNames(std::move(roles));
    EXPECT_EQ(resp.RoleNames().size(), 2);
}

class DescribeUserResponseTest : public ::testing::Test {};

TEST_F(DescribeUserResponseTest, SetterAndGetter) {
    milvus::DescribeUserResponse resp;
    milvus::UserDesc desc;
    resp.SetDesc(std::move(desc));
    (void)resp.Desc();
}

class DescribeRoleResponseTest : public ::testing::Test {};

TEST_F(DescribeRoleResponseTest, SetterAndGetter) {
    milvus::DescribeRoleResponse resp;
    milvus::RoleDesc desc;
    resp.SetDesc(std::move(desc));
    (void)resp.Desc();
}

class ListPrivilegeGroupsResponseTest : public ::testing::Test {};

TEST_F(ListPrivilegeGroupsResponseTest, SetterAndGetter) {
    milvus::ListPrivilegeGroupsResponse resp;
    milvus::PrivilegeGroupInfos groups;
    resp.SetGroups(std::move(groups));
    (void)resp.Groups();
}
