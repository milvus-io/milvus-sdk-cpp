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

class CreateUserRequestTest : public ::testing::Test {};

TEST_F(CreateUserRequestTest, GettersAndSetters) {
    milvus::CreateUserRequest req;

    req.WithUserName("admin");
    EXPECT_EQ(req.UserName(), "admin");

    req.WithPassword("secret123");
    EXPECT_EQ(req.Password(), "secret123");
}

class UpdatePasswordRequestTest : public ::testing::Test {};

TEST_F(UpdatePasswordRequestTest, GettersAndSetters) {
    milvus::UpdatePasswordRequest req;

    req.WithUserName("admin");
    EXPECT_EQ(req.UserName(), "admin");

    req.WithOldPassword("old_pass");
    EXPECT_EQ(req.OldPassword(), "old_pass");

    req.WithNewPassword("new_pass");
    EXPECT_EQ(req.NewPassword(), "new_pass");
}

class DropUserRequestTest : public ::testing::Test {};

TEST_F(DropUserRequestTest, GettersAndSetters) {
    milvus::DropUserRequest req;
    req.WithUserName("drop_user");
    EXPECT_EQ(req.UserName(), "drop_user");
}

class DescribeUserRequestTest : public ::testing::Test {};

TEST_F(DescribeUserRequestTest, GettersAndSetters) {
    milvus::DescribeUserRequest req;
    req.WithUserName("desc_user");
    EXPECT_EQ(req.UserName(), "desc_user");
}

class CreateRoleRequestTest : public ::testing::Test {};

TEST_F(CreateRoleRequestTest, GettersAndSetters) {
    milvus::CreateRoleRequest req;
    req.WithRoleName("admin_role");
    EXPECT_EQ(req.RoleName(), "admin_role");
}

class DropRoleRequestTest : public ::testing::Test {};

TEST_F(DropRoleRequestTest, GettersAndSetters) {
    milvus::DropRoleRequest req;

    req.WithRoleName("drop_role");
    EXPECT_EQ(req.RoleName(), "drop_role");

    req.WithForceDrop(true);
    EXPECT_TRUE(req.ForceDrop());

    req.WithForceDrop(false);
    EXPECT_FALSE(req.ForceDrop());
}

class DescribeRoleRequestTest : public ::testing::Test {};

TEST_F(DescribeRoleRequestTest, GettersAndSetters) {
    milvus::DescribeRoleRequest req;

    req.WithRoleName("desc_role");
    EXPECT_EQ(req.RoleName(), "desc_role");

    req.WithDatabaseName("role_db");
    EXPECT_EQ(req.DatabaseName(), "role_db");
}

class GrantRoleRequestTest : public ::testing::Test {};

TEST_F(GrantRoleRequestTest, GettersAndSetters) {
    milvus::GrantRoleRequest req;

    req.WithUserName("admin");
    EXPECT_EQ(req.UserName(), "admin");

    req.WithRoleName("admin_role");
    EXPECT_EQ(req.RoleName(), "admin_role");
}

class RevokeRoleRequestTest : public ::testing::Test {};

TEST_F(RevokeRoleRequestTest, GettersAndSetters) {
    milvus::RevokeRoleRequest req;

    req.WithUserName("admin");
    EXPECT_EQ(req.UserName(), "admin");

    req.WithRoleName("admin_role");
    EXPECT_EQ(req.RoleName(), "admin_role");
}

class GrantPrivilegeV2RequestTest : public ::testing::Test {};

TEST_F(GrantPrivilegeV2RequestTest, GettersAndSetters) {
    milvus::GrantPrivilegeV2Request req;

    req.WithRoleName("admin_role");
    EXPECT_EQ(req.RoleName(), "admin_role");

    req.WithDatabaseName("priv_db");
    EXPECT_EQ(req.DatabaseName(), "priv_db");

    req.WithCollectionName("priv_coll");
    EXPECT_EQ(req.CollectionName(), "priv_coll");

    req.WithPrivilege("Insert");
    EXPECT_EQ(req.Privilege(), "Insert");
}

class RevokePrivilegeV2RequestTest : public ::testing::Test {};

TEST_F(RevokePrivilegeV2RequestTest, GettersAndSetters) {
    milvus::RevokePrivilegeV2Request req;

    req.WithRoleName("admin_role");
    EXPECT_EQ(req.RoleName(), "admin_role");

    req.WithDatabaseName("priv_db");
    EXPECT_EQ(req.DatabaseName(), "priv_db");

    req.WithCollectionName("priv_coll");
    EXPECT_EQ(req.CollectionName(), "priv_coll");

    req.WithPrivilege("Search");
    EXPECT_EQ(req.Privilege(), "Search");
}

class CreatePrivilegeGroupRequestTest : public ::testing::Test {};

TEST_F(CreatePrivilegeGroupRequestTest, GettersAndSetters) {
    milvus::CreatePrivilegeGroupRequest req;
    req.WithGroupName("group1");
    EXPECT_EQ(req.GroupName(), "group1");
}

class DropPrivilegeGroupRequestTest : public ::testing::Test {};

TEST_F(DropPrivilegeGroupRequestTest, GettersAndSetters) {
    milvus::DropPrivilegeGroupRequest req;
    req.WithGroupName("group1");
    EXPECT_EQ(req.GroupName(), "group1");
}

class AddPrivilegesToGroupRequestTest : public ::testing::Test {};

TEST_F(AddPrivilegesToGroupRequestTest, GettersAndSetters) {
    milvus::AddPrivilegesToGroupRequest req;

    req.WithGroupName("group1");
    EXPECT_EQ(req.GroupName(), "group1");

    req.AddPrivilege("Insert");
    req.AddPrivilege("Search");
    EXPECT_EQ(req.Privileges().size(), 2);
    EXPECT_TRUE(req.Privileges().count("Insert"));
    EXPECT_TRUE(req.Privileges().count("Search"));
}

class RemovePrivilegesFromGroupRequestTest : public ::testing::Test {};

TEST_F(RemovePrivilegesFromGroupRequestTest, GettersAndSetters) {
    milvus::RemovePrivilegesFromGroupRequest req;

    req.WithGroupName("group2");
    EXPECT_EQ(req.GroupName(), "group2");

    req.AddPrivilege("Delete");
    EXPECT_EQ(req.Privileges().size(), 1);
    EXPECT_TRUE(req.Privileges().count("Delete"));
}

TEST_F(CreateRoleRequestTest, SetRoleName) {
    milvus::CreateRoleRequest req;
    req.SetRoleName("my_role");
    EXPECT_EQ(req.RoleName(), "my_role");
}

TEST_F(DropRoleRequestTest, SetRoleNameAndForceDrop) {
    milvus::DropRoleRequest req;
    req.SetRoleName("role_to_drop");
    EXPECT_EQ(req.RoleName(), "role_to_drop");

    req.SetForceDrop(true);
    EXPECT_TRUE(req.ForceDrop());
}

TEST_F(CreatePrivilegeGroupRequestTest, SetGroupName) {
    milvus::CreatePrivilegeGroupRequest req;
    req.SetGroupName("pg_set");
    EXPECT_EQ(req.GroupName(), "pg_set");
}

TEST_F(DropPrivilegeGroupRequestTest, SetGroupName) {
    milvus::DropPrivilegeGroupRequest req;
    req.SetGroupName("pg_drop");
    EXPECT_EQ(req.GroupName(), "pg_drop");
}

TEST_F(AddPrivilegesToGroupRequestTest, SetGroupNameAndPrivileges) {
    milvus::AddPrivilegesToGroupRequest req;
    req.SetGroupName("pg_add");
    EXPECT_EQ(req.GroupName(), "pg_add");

    std::set<std::string> privs = {"Insert", "Search"};
    req.SetPrivileges(std::move(privs));
    EXPECT_EQ(req.Privileges().size(), 2);
    EXPECT_TRUE(req.Privileges().count("Insert"));

    std::set<std::string> privs2 = {"Delete"};
    auto& ref = req.WithPrivileges(std::move(privs2));
    EXPECT_EQ(req.Privileges().size(), 1);
    EXPECT_TRUE(req.Privileges().count("Delete"));
    EXPECT_EQ(&ref, &req);
}

TEST_F(RemovePrivilegesFromGroupRequestTest, SetGroupNameAndPrivileges) {
    milvus::RemovePrivilegesFromGroupRequest req;
    req.SetGroupName("pg_remove");
    EXPECT_EQ(req.GroupName(), "pg_remove");

    std::set<std::string> privs = {"Insert"};
    req.SetPrivileges(std::move(privs));
    EXPECT_EQ(req.Privileges().size(), 1);

    std::set<std::string> privs2 = {"Delete", "Query"};
    auto& ref = req.WithPrivileges(std::move(privs2));
    EXPECT_EQ(req.Privileges().size(), 2);
    EXPECT_EQ(&ref, &req);
}
