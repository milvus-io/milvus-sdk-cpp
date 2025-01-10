#pragma once

#include <string>
#include <vector>

class PrivilegeGroupInfo {
public:
    PrivilegeGroupInfo(const std::string& group_name);

    void AddPrivilege(const std::string& privilege);
    const std::string& GroupName() const;
    void SetGroupName(const std::string& group_name);
    const std::vector<std::string>& Privileges() const;
    void SetPrivileges(const std::vector<std::string>& privileges);

private:
    std::string group_name;
    std::vector<std::string> privileges;
};
