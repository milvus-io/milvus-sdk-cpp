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
#include "milvus/MilvusClient.h"

std::vector<std::string>
ListPrivilegeGroups(milvus::MilvusClientPtr& client) {
    milvus::PrivilegeGroupInfos groups;
    auto status = client->ListPrivilegeGroups(groups);
    util::CheckStatus("list privilege groups", status);
    std::vector<std::string> names;
    for (auto& group : groups) {
        names.push_back(group.Name());
    }
    util::PrintList(names);
    return names;
}

std::vector<std::string>
ListRoles(milvus::MilvusClientPtr& client) {
    std::vector<std::string> roles;
    auto status = client->ListRoles(roles);
    util::CheckStatus("list roles", status);
    util::PrintList(roles);
    return roles;
}

std::vector<std::string>
ListUsers(milvus::MilvusClientPtr& client) {
    std::vector<std::string> users;
    auto status = client->ListUsers(users);
    util::CheckStatus("list users", status);
    util::PrintList(users);
    return users;
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

    auto client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    // create a collection
    const std::string collection_name = "my_rbac_collection";
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({"pk", milvus::DataType::INT64, "", true, true});
    collection_schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(8));

    status = client->CreateCollection(collection_schema);
    util::CheckStatus("create collection: " + collection_name, status);

    milvus::IndexDesc index_vector("vector", "", milvus::IndexType::AUTOINDEX, milvus::MetricType::L2);
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("create index on vector field", status);

    status = client->LoadCollection(collection_name);
    util::CheckStatus("load collection: " + collection_name, status);

    ListRoles(client);
    ListUsers(client);

    const std::string role_name = "my_new_role";
    const std::string user_name = "my_new_user";
    const std::string privilege_group_name = "my_privilege_group";

    // new privilege group
    client->DropPrivilegeGroup(privilege_group_name);
    status = client->CreatePrivilegeGroup(privilege_group_name);
    util::CheckStatus("create privilege group: " + privilege_group_name, status);

    std::vector<std::string> privileges = {"Search", "Query"};
    status = client->AddPrivilegesToGroup(privilege_group_name, privileges);
    util::CheckStatus("add privileges to group: " + privilege_group_name, status);

    // new role
    client->DropRole(role_name, true);
    status = client->CreateRole(role_name);
    util::CheckStatus("create role: " + role_name, status);

    status = client->GrantPrivilege(role_name, privilege_group_name, collection_name, "default");
    util::CheckStatus("grant privilege group to role: " + role_name, status);

    milvus::RoleDesc role_desc;
    status = client->DescribeRole(role_name, role_desc);
    util::CheckStatus("describe role: " + role_name, status);
    PrintRole(role_desc);

    // new user
    client->DropUser(user_name);
    status = client->CreateUser(user_name, "aaaaaa");
    util::CheckStatus("create user: " + user_name, status);

    status = client->UpdatePassword(user_name, "aaaaaa", "123456");
    util::CheckStatus("update password for user: " + user_name, status);

    client->GrantRole(user_name, role_name);
    util::CheckStatus("grant role: " + role_name + " to user: " + user_name, status);

    milvus::UserDesc user_desc;
    status = client->DescribeUser(user_name, user_desc);
    util::CheckStatus("describe user: " + user_name, status);
    PrintUser(user_desc);

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

    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", rows, dml_results);
    if (status.IsOk()) {
        std::cout << "UNEXPECTED! Insert is expected to fail but it succeed" << std::endl;
    } else {
        std::cout << "Insert failed with error: " << status.Message() << std::endl;
    }

    // query is ok
    milvus::QueryArguments q_count{};
    q_count.SetCollectionName(collection_name);
    q_count.AddOutputField("count(*)");

    milvus::QueryResults count_result{};
    status = client->Query(q_count, count_result);
    util::CheckStatus("query count(*)", status);
    std::cout << "count(*) = " << count_result.GetRowCount() << std::endl;

    // connect with root to drop the user/role/privilege_group
    client->Disconnect();
    connect_param.SetAuthorizations("root", "Milvus");
    status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server with root", status);

    status = client->RemovePrivilegesFromGroup(privilege_group_name, privileges);
    util::CheckStatus("remove privileges from group: " + privilege_group_name, status);

    status = client->RevokePrivilege(role_name, privilege_group_name, collection_name, "default");
    util::CheckStatus("revoke privilege group from role: " + role_name, status);

    status = client->RevokeRole(user_name, role_name);
    util::CheckStatus("revoke role from user: " + user_name, status);

    status = client->DropUser(user_name);
    util::CheckStatus("drop user: " + user_name, status);

    status = client->DropRole(role_name);
    util::CheckStatus("drop role: " + role_name, status);

    status = client->DropPrivilegeGroup(role_name);
    util::CheckStatus("drop privilege group: " + privilege_group_name, status);

    client->Disconnect();
    return 0;
}
