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

#pragma once

#include <string>
#include <vector>

namespace milvus {

/**
 * @brief User description. Used by MilvusClient::DescribeUser().
 */
class UserDesc {
 public:
    /**
     * @brief Construct a new UserDesc object.
     */
    UserDesc();

    /**
     * @brief Construct a new UserDesc object.
     */
    UserDesc(const std::string& name, std::vector<std::string>&& roles);

    /**
     * @brief Set the name of the user.
     */
    void
    SetName(const std::string& name);

    /**
     * @brief Get the name of the user.
     */
    const std::string&
    Name() const;

    /**
     * @brief Add a role name for the user.
     */
    void
    AddRole(const std::string& role_name);

    /**
     * @brief Get role names of the user.
     */
    const std::vector<std::string>&
    Roles() const;

 private:
    std::string name_;
    std::vector<std::string> roles_;
};

}  // namespace milvus
