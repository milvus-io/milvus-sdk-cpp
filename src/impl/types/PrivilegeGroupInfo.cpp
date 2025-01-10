#include "milvus/types/PrivilegeGroupInfo.h"

PrivilegeGroupInfo::PrivilegeGroupInfo(const std::string& group_name) : group_name(group_name) {}

void PrivilegeGroupInfo::AddPrivilege(const std::string& privilege) {
    privileges.push_back(privilege);
}

const std::string& PrivilegeGroupInfo::GroupName() const {
    return group_name;
}

void PrivilegeGroupInfo::SetGroupName(const std::string& group_name) {
    this->group_name = group_name;
}

const std::vector<std::string>& PrivilegeGroupInfo::Privileges() const {
    return privileges;
}

void PrivilegeGroupInfo::SetPrivileges(const std::vector<std::string>& privileges) {
    this->privileges = privileges;
}
