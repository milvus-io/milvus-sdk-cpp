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

#include <algorithm>

#include "MilvusServerTest.h"

using milvus::test::MilvusServerTest;

class MilvusServerTestRbac : public MilvusServerTest {};

TEST_F(MilvusServerTestRbac, UserCRUD) {
    std::string username = milvus::test::RanName("user_");
    std::string password = "Test1234!";

    // create user
    auto status = client_->CreateUser(milvus::CreateUserRequest().WithUserName(username).WithPassword(password));
    milvus::test::ExpectStatusOK(status);

    // list users
    milvus::ListUsersResponse list_resp;
    status = client_->ListUsers(milvus::ListUsersRequest(), list_resp);
    milvus::test::ExpectStatusOK(status);
    auto& names = list_resp.UserNames();
    EXPECT_NE(std::find(names.begin(), names.end(), username), names.end());

    // describe user
    milvus::DescribeUserResponse desc_resp;
    status = client_->DescribeUser(milvus::DescribeUserRequest().WithUserName(username), desc_resp);
    milvus::test::ExpectStatusOK(status);

    // update password
    std::string new_password = "NewPass5678!";
    status = client_->UpdatePassword(
        milvus::UpdatePasswordRequest().WithUserName(username).WithOldPassword(password).WithNewPassword(new_password));
    milvus::test::ExpectStatusOK(status);

    // drop user
    status = client_->DropUser(milvus::DropUserRequest().WithUserName(username));
    milvus::test::ExpectStatusOK(status);

    // verify dropped
    milvus::ListUsersResponse list_resp2;
    status = client_->ListUsers(milvus::ListUsersRequest(), list_resp2);
    milvus::test::ExpectStatusOK(status);
    auto& names2 = list_resp2.UserNames();
    EXPECT_EQ(std::find(names2.begin(), names2.end(), username), names2.end());
}

TEST_F(MilvusServerTestRbac, RoleCRUD) {
    std::string role_name = milvus::test::RanName("role_");

    // create role
    auto status = client_->CreateRole(milvus::CreateRoleRequest().WithRoleName(role_name));
    milvus::test::ExpectStatusOK(status);

    // list roles
    milvus::ListRolesResponse list_resp;
    status = client_->ListRoles(milvus::ListRolesRequest(), list_resp);
    milvus::test::ExpectStatusOK(status);
    auto& roles = list_resp.RoleNames();
    EXPECT_NE(std::find(roles.begin(), roles.end(), role_name), roles.end());

    // describe role
    milvus::DescribeRoleResponse desc_resp;
    status = client_->DescribeRole(milvus::DescribeRoleRequest().WithRoleName(role_name), desc_resp);
    milvus::test::ExpectStatusOK(status);

    // drop role
    status = client_->DropRole(milvus::DropRoleRequest().WithRoleName(role_name));
    milvus::test::ExpectStatusOK(status);

    // verify dropped
    milvus::ListRolesResponse list_resp2;
    status = client_->ListRoles(milvus::ListRolesRequest(), list_resp2);
    milvus::test::ExpectStatusOK(status);
    auto& roles2 = list_resp2.RoleNames();
    EXPECT_EQ(std::find(roles2.begin(), roles2.end(), role_name), roles2.end());
}

TEST_F(MilvusServerTestRbac, GrantAndRevokeRole) {
    std::string username = milvus::test::RanName("user_");
    std::string role_name = milvus::test::RanName("role_");

    // create user and role
    auto status = client_->CreateUser(milvus::CreateUserRequest().WithUserName(username).WithPassword("Test1234!"));
    milvus::test::ExpectStatusOK(status);
    status = client_->CreateRole(milvus::CreateRoleRequest().WithRoleName(role_name));
    milvus::test::ExpectStatusOK(status);

    // grant role to user
    status = client_->GrantRole(milvus::GrantRoleRequest().WithUserName(username).WithRoleName(role_name));
    milvus::test::ExpectStatusOK(status);

    // verify user has the role
    milvus::DescribeUserResponse desc_resp;
    status = client_->DescribeUser(milvus::DescribeUserRequest().WithUserName(username), desc_resp);
    milvus::test::ExpectStatusOK(status);

    // revoke role from user
    status = client_->RevokeRole(milvus::RevokeRoleRequest().WithUserName(username).WithRoleName(role_name));
    milvus::test::ExpectStatusOK(status);

    // cleanup
    client_->DropUser(milvus::DropUserRequest().WithUserName(username));
    client_->DropRole(milvus::DropRoleRequest().WithRoleName(role_name));
}

// // this case can only run on milvus with authorizationEnabled=true
// TEST_F(MilvusServerTestRbac, GrantAndRevokePrivilege) {
//     std::string role_name = milvus::test::RanName("role_");

//     std::cout << "Testing grant and revoke privilege for role: " << role_name << std::endl;

//     // create role
//     auto status = client_->CreateRole(milvus::CreateRoleRequest().WithRoleName(role_name));
//     milvus::test::ExpectStatusOK(status);

//     std::cout << "Created role: " << role_name << std::endl;

//     // grant privilege to role
//     status = client_->GrantPrivilegeV2(milvus::GrantPrivilegeV2Request()
//                                            .WithRoleName(role_name)
//                                            .WithPrivilege("Search")
//                                            .WithCollectionName("*")
//                                            .WithDatabaseName("*"));
//     milvus::test::ExpectStatusOK(status);

//     std::cout << "Granted Search privilege to role: " << role_name << std::endl;

//     // describe role to verify privilege
//     milvus::DescribeRoleResponse desc_resp;
//     status = client_->DescribeRole(milvus::DescribeRoleRequest().WithRoleName(role_name), desc_resp);
//     milvus::test::ExpectStatusOK(status);

//     std::cout << "Role description after granting privilege " << std::endl;

//     // revoke privilege from role
//     status = client_->RevokePrivilegeV2(milvus::RevokePrivilegeV2Request()
//                                             .WithRoleName(role_name)
//                                             .WithPrivilege("Search")
//                                             .WithCollectionName("*")
//                                             .WithDatabaseName("*"));
//     milvus::test::ExpectStatusOK(status);

//     std::cout << "Revoked Search privilege from role: " << role_name << std::endl;

//     // cleanup
//     client_->DropRole(milvus::DropRoleRequest().WithRoleName(role_name));
// }

TEST_F(MilvusServerTestRbac, PrivilegeGroupCRUD) {
    std::string group_name = milvus::test::RanName("privgrp_");

    // create privilege group
    auto status = client_->CreatePrivilegeGroup(milvus::CreatePrivilegeGroupRequest().WithGroupName(group_name));
    milvus::test::ExpectStatusOK(status);

    // list privilege groups
    milvus::ListPrivilegeGroupsResponse list_resp;
    status = client_->ListPrivilegeGroups(milvus::ListPrivilegeGroupsRequest(), list_resp);
    milvus::test::ExpectStatusOK(status);

    // add privileges to group
    status = client_->AddPrivilegesToGroup(
        milvus::AddPrivilegesToGroupRequest().WithGroupName(group_name).AddPrivilege("Search").AddPrivilege("Query"));
    milvus::test::ExpectStatusOK(status);

    // remove privileges from group
    status = client_->RemovePrivilegesFromGroup(
        milvus::RemovePrivilegesFromGroupRequest().WithGroupName(group_name).AddPrivilege("Query"));
    milvus::test::ExpectStatusOK(status);

    // drop privilege group
    status = client_->DropPrivilegeGroup(milvus::DropPrivilegeGroupRequest().WithGroupName(group_name));
    milvus::test::ExpectStatusOK(status);
}
