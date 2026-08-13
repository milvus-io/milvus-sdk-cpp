#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

#include "milvus/MilvusClientV2.h"
namespace {
const char*
Env(const char* n, const char* d) {
    const char* v = std::getenv(n);
    return v ? v : d;
}
bool
Ok(const milvus::Status& s, const std::string& op) {
    if (s.IsOk()) {
        return true;
    }
    std::cerr << "Failed to " << op << ": " << s.Message() << std::endl;
    return false;
}
}  // namespace
int
main() {
    auto client = milvus::MilvusClientV2::Create();
    // Connect with an administrator credential allowed to create users, roles, privilege groups,
    // and grants.
    std::cout << "Calling Connect..." << std::endl;
    auto status = client->Connect(
        milvus::ConnectParam{Env("MILVUS_URI", "http://localhost:19530"), Env("MILVUS_TOKEN", "root:Milvus")});
    if (!Ok(status, "connect")) {
        return 1;
    }
    std::cout << "Connect succeeded." << std::endl;
    const std::string user = "cpp_tutorial_user", role = "cpp_tutorial_role", group = "cpp_tutorial_group";
    std::cout << "Calling DropUser for stale data..." << std::endl;
    status = client->DropUser(milvus::DropUserRequest().WithUserName(user));
    if (!Ok(status, "drop stale user")) {
        return 1;
    }
    std::cout << "DropUser stale-data cleanup completed." << std::endl;
    std::cout << "Calling DropRole for stale data..." << std::endl;
    status = client->DropRole(milvus::DropRoleRequest().WithRoleName(role).WithForceDrop(true));
    if (!Ok(status, "drop stale role")) {
        return 1;
    }
    std::cout << "DropRole stale-data cleanup completed." << std::endl;
    std::cout << "Calling DropPrivilegeGroup for stale data..." << std::endl;
    status = client->DropPrivilegeGroup(milvus::DropPrivilegeGroupRequest().WithGroupName(group));
    if (!Ok(status, "drop stale privilege group")) {
        return 1;
    }
    std::cout << "DropPrivilegeGroup stale-data cleanup completed." << std::endl;
    // CreateUser creates a login identity with the supplied username and password.
    std::cout << "Calling CreateUser..." << std::endl;
    status = client->CreateUser(
        milvus::CreateUserRequest().WithUserName(user).WithPassword(Env("MILVUS_USER_PASSWORD", "CppTutorial!123")));
    if (!Ok(status, "create user")) {
        return 1;
    }
    std::cout << "CreateUser succeeded." << std::endl;
    // CreateRole creates a named permission container; privileges are attached in later calls.
    std::cout << "Calling CreateRole..." << std::endl;
    status = client->CreateRole(milvus::CreateRoleRequest().WithRoleName(role));
    if (!Ok(status, "create role")) {
        return 1;
    }
    std::cout << "CreateRole succeeded." << std::endl;
    // CreatePrivilegeGroup creates a reusable custom group identified by its group name.
    std::cout << "Calling CreatePrivilegeGroup..." << std::endl;
    status = client->CreatePrivilegeGroup(milvus::CreatePrivilegeGroupRequest().WithGroupName(group));
    if (!Ok(status, "create privilege group")) {
        return 1;
    }
    std::cout << "CreatePrivilegeGroup succeeded." << std::endl;
    // AddPrivilegesToGroup adds built-in privileges to the group. Query permits scalar reads, and
    // Search permits vector searches.
    std::cout << "Calling AddPrivilegesToGroup..." << std::endl;
    status = client->AddPrivilegesToGroup(milvus::AddPrivilegesToGroupRequest().WithGroupName(group).WithPrivileges(
        std::set<std::string>{"Query", "Search"}));
    if (!Ok(status, "add privileges")) {
        return 1;
    }
    std::cout << "AddPrivilegesToGroup succeeded." << std::endl;
    // GrantPrivilegeV2 attaches the custom group to the role. The database selects the scope,
    // collection "*" covers all collections, and privilege names the group.
    std::cout << "Calling GrantPrivilegeV2..." << std::endl;
    status = client->GrantPrivilegeV2(milvus::GrantPrivilegeV2Request()
                                          .WithRoleName(role)
                                          .WithDatabaseName("default")
                                          .WithCollectionName("*")
                                          .WithPrivilege(group));
    if (!Ok(status, "grant privilege")) {
        return 1;
    }
    std::cout << "GrantPrivilegeV2 succeeded." << std::endl;
    // GrantRole assigns the named role to the named user.
    std::cout << "Calling GrantRole..." << std::endl;
    status = client->GrantRole(milvus::GrantRoleRequest().WithUserName(user).WithRoleName(role));
    if (!Ok(status, "grant role")) {
        return 1;
    }
    std::cout << "GrantRole succeeded." << std::endl;
    milvus::DescribeUserResponse user_response;
    // DescribeUser returns user metadata and the role names assigned to the user.
    std::cout << "Calling DescribeUser..." << std::endl;
    status = client->DescribeUser(milvus::DescribeUserRequest().WithUserName(user), user_response);
    if (!Ok(status, "describe user")) {
        return 1;
    }
    std::cout << "DescribeUser succeeded." << std::endl;
    std::cout << "User: " << user_response.Desc().Name() << ", roles=" << user_response.Desc().Roles().size()
              << std::endl;
    // DropUser permanently removes the temporary login identity.
    std::cout << "Calling DropUser..." << std::endl;
    status = client->DropUser(milvus::DropUserRequest().WithUserName(user));
    if (!Ok(status, "drop user")) {
        return 1;
    }
    std::cout << "DropUser succeeded." << std::endl;
    // DropRole removes the temporary role; force drop also clears remaining associations.
    std::cout << "Calling DropRole..." << std::endl;
    status = client->DropRole(milvus::DropRoleRequest().WithRoleName(role).WithForceDrop(true));
    if (!Ok(status, "drop role")) {
        return 1;
    }
    std::cout << "DropRole succeeded." << std::endl;
    // DropPrivilegeGroup permanently removes the custom privilege group.
    std::cout << "Calling DropPrivilegeGroup..." << std::endl;
    status = client->DropPrivilegeGroup(milvus::DropPrivilegeGroupRequest().WithGroupName(group));
    if (!Ok(status, "drop privilege group")) {
        return 1;
    }
    std::cout << "DropPrivilegeGroup succeeded." << std::endl;
    return 0;
}
