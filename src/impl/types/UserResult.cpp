#include "milvus/types/UserResult.h"

namespace milvus {

UserResult::UserResult() = default;

void UserResult::SetUserName(const std::string& user_name) {
    user_name_ = user_name;
}

const std::string& UserResult::UserName() const {
    return user_name_;
}

void UserResult::AddRole(const std::string& role) {
    roles_.emplace_back(role);
}

const std::vector<std::string>& UserResult::Roles() const {
    return roles_;
}

}  // namespace milvus 