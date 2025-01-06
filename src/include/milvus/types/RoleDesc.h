#pragma once

#include <string>
#include <vector>

namespace milvus {

struct Privilege {
    std::string object_type;
    std::string object_name;
    std::string db_name;
    std::string role_name;
    std::string privilege;
    std::string grantor_name;
};

class RoleDesc {
public:
    RoleDesc();
    RoleDesc(const std::string& role, const std::vector<Privilege>& privileges);

    const std::string& GetRole() const;
    const std::vector<Privilege>& GetPrivileges() const;

private:
    std::string role_;
    std::vector<Privilege> privileges_;
};

}  // namespace milvus
