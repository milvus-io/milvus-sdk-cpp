#pragma once

#include <string>
#include <vector>

namespace milvus {

class UserResult {
public:
    UserResult();

    void SetUserName(const std::string& user_name);
    const std::string& UserName() const;

    void AddRole(const std::string& role);
    const std::vector<std::string>& Roles() const;

private:
    std::string user_name_;
    std::vector<std::string> roles_;
};

}  // namespace milvus 