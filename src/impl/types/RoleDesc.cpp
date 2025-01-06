#include "milvus/types/RoleDesc.h"

namespace milvus {

RoleDesc::RoleDesc() {}

RoleDesc::RoleDesc(const std::string& role, const std::vector<Privilege>& privileges)
    : role_(role), privileges_(privileges) {}

const std::string& RoleDesc::GetRole() const {
    return role_;
}

const std::vector<Privilege>& RoleDesc::GetPrivileges() const {
    return privileges_;
}

}  // namespace milvus
