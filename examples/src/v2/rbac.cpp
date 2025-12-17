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

#include <iostream>
#include <string>
#include <thread>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

std::vector<std::string>
ListPrivilegeGroups(milvus::MilvusClientV2Ptr& client) {
    milvus::ListPrivilegeGroupsRequest request;
    milvus::ListPrivilegeGroupsResponse response;
    auto status = client->ListPrivilegeGroups(request, response);
    util::CheckStatus("list privilege groups", status);
    std::vector<std::string> names;
    for (auto& group : response.Groups()) {
        names.push_back(group.Name());
    }
    util::PrintList(names);
    return names;
}

std::vector<std::string>
ListRoles(milvus::MilvusClientV2Ptr& client) {
    milvus::ListRolesRequest request;
    milvus::ListRolesResponse response;
    auto status = client->ListRoles(request, response);
    util::CheckStatus("list roles", status);
    util::PrintList(response.RoleNames());
    return response.RoleNames();
}

std::vector<std::string>
ListUsers(milvus::MilvusClientV2Ptr& client) {
    milvus::ListUsersRequest request;
    milvus::ListUsersResponse response;
    auto status = client->ListUsers(request, response);
    util::CheckStatus("list users", status);
    util::PrintList(response.UserNames());
    return response.UserNames();
}

void
PrintRole(const milvus::RoleDesc& role_desc) {
    std::cout << "Role '" + role_desc.Name() << "' privileges:" << std::endl;
    for (const auto& item : role_desc.GrantItems()) {
        std::cout << "{object:" << item.object_type_ << ", object_name:" << item.object_name_
                  << ", db_name:" << item.db_name_ << ", grantor_name:" << item.grantor_name_
                  << ", privilege:" << item.privilege_ << "}" << std::endl;
    }
}

void
PrintUser(const milvus::UserDesc& user_desc) {
    std::cout << "User '" + user_desc.Name() << "' roles:" << std::endl;
    util::PrintList(user_desc.Roles());
    std::cout << std::endl;
}

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    // create a collection
    const std::string collection_name = "CPP_V2_RBAC";
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>(collection_name);
    collection_schema->AddField({"pk", milvus::DataType::INT64, "", true, true});
    collection_schema->AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(8));

    status = client->CreateCollection(milvus::CreateCollectionRequest().WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    milvus::IndexDesc index_vector("vector", "", milvus::IndexType::AUTOINDEX, milvus::MetricType::L2);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + collection_name, status);

    ListRoles(client);
    ListUsers(client);

    const std::string role_name = "my_new_role";
    const std::string user_name = "my_new_user";
    const std::string privilege_group_name = "my_privilege_group";

    // new privilege group
    client->DropPrivilegeGroup(milvus::DropPrivilegeGroupRequest().WithGroupName(privilege_group_name));
    status = client->CreatePrivilegeGroup(milvus::CreatePrivilegeGroupRequest().WithGroupName(privilege_group_name));
    util::CheckStatus("create privilege group: " + privilege_group_name, status);

    std::set<std::string> privileges = {"Search", "Query"};
    status = client->AddPrivilegesToGroup(milvus::AddPrivilegesToGroupRequest()
                                              .WithGroupName(privilege_group_name)
                                              .WithPrivileges(std::move(privileges)));
    util::CheckStatus("add privileges to group: " + privilege_group_name, status);

    // new role
    client->DropRole(milvus::DropRoleRequest().WithRoleName(role_name).WithForceDrop(true));
    status = client->CreateRole(milvus::CreateRoleRequest().WithRoleName(role_name));
    util::CheckStatus("create role: " + role_name, status);

    status = client->GrantPrivilegeV2(milvus::GrantPrivilegeV2Request()
                                          .WithRoleName(role_name)
                                          .WithPrivilege(privilege_group_name)
                                          .WithCollectionName(collection_name));
    util::CheckStatus("grant privilege group to role: " + role_name, status);

    milvus::DescribeRoleResponse resp_desc_role;
    status = client->DescribeRole(milvus::DescribeRoleRequest().WithRoleName(role_name), resp_desc_role);
    util::CheckStatus("describe role: " + role_name, status);
    PrintRole(resp_desc_role.Desc());

    // new user
    client->DropUser(milvus::DropUserRequest().WithUserName(user_name));
    status = client->CreateUser(milvus::CreateUserRequest().WithUserName(user_name).WithPassword("aaaaaa"));
    util::CheckStatus("create user: " + user_name, status);

    status = client->UpdatePassword(
        milvus::UpdatePasswordRequest().WithUserName(user_name).WithOldPassword("aaaaaa").WithNewPassword("123456"));
    util::CheckStatus("update password for user: " + user_name, status);

    client->GrantRole(milvus::GrantRoleRequest().WithUserName(user_name).WithRoleName(role_name));
    util::CheckStatus("grant role: " + role_name + " to user: " + user_name, status);

    milvus::DescribeUserResponse resp_desc_user;
    status = client->DescribeUser(milvus::DescribeUserRequest().WithUserName(user_name), resp_desc_user);
    util::CheckStatus("describe user: " + user_name, status);
    PrintUser(resp_desc_user.Desc());

    ListPrivilegeGroups(client);
    ListRoles(client);
    ListUsers(client);

    // connect with new user
    client->Disconnect();
    connect_param.SetAuthorizations(user_name, "123456");
    status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server with user: " + user_name, status);

    // this user has no privilege to insert data, this call is expected to fail
    milvus::EntityRows rows;
    milvus::EntityRow row;
    row["vector"] = util::GenerateFloatVector(8);
    rows.emplace_back(std::move(row));

    milvus::InsertResponse resp_insert;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            resp_insert);
    if (status.IsOk()) {
        std::cout << "UNEXPECTED! Insert is expected to fail but it succeed" << std::endl;
    } else {
        std::cout << "Insert failed with error: " << status.Message() << std::endl;
    }

    {
        // query is ok
        auto request = milvus::QueryRequest().WithCollectionName(collection_name).AddOutputField("count(*)");

        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query count(*)", status);
        std::cout << "count(*) = " << response.Results().GetRowCount() << std::endl;
    }

    // connect with root to drop the user/role/privilege_group
    client->Disconnect();
    connect_param.SetAuthorizations("root", "Milvus");
    status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server with root", status);

    privileges = {"Search", "Query"};
    status = client->RemovePrivilegesFromGroup(milvus::RemovePrivilegesFromGroupRequest()
                                                   .WithGroupName(privilege_group_name)
                                                   .WithPrivileges(std::move(privileges)));
    util::CheckStatus("remove privileges from group: " + privilege_group_name, status);

    status = client->RevokePrivilegeV2(milvus::RevokePrivilegeV2Request()
                                           .WithRoleName(role_name)
                                           .WithPrivilege(privilege_group_name)
                                           .WithCollectionName(collection_name));
    util::CheckStatus("revoke privilege group from role: " + role_name, status);

    status = client->RevokeRole(milvus::RevokeRoleRequest().WithUserName(user_name).WithRoleName(role_name));
    util::CheckStatus("revoke role from user: " + user_name, status);

    status = client->DropUser(milvus::DropUserRequest().WithUserName(user_name));
    util::CheckStatus("drop user: " + user_name, status);

    status = client->DropRole(milvus::DropRoleRequest().WithRoleName(role_name).WithForceDrop(false));
    util::CheckStatus("drop role: " + role_name, status);

    status = client->DropPrivilegeGroup(milvus::DropPrivilegeGroupRequest().WithGroupName(privilege_group_name));
    util::CheckStatus("drop privilege group: " + privilege_group_name, status);

    client->Disconnect();
    return 0;
}
