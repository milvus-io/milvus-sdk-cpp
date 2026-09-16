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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief User description. Used by MilvusClient::DescribeUser().
 */
class MILVUS_SDK_API UserDesc {
 public:
    /**
     * @brief Construct a new UserDesc object.
     */
    UserDesc();

    /**
     * @brief Construct a new UserDesc object.
     * @param [in] name the name.
     * @param [in] roles the roles.
     */
    UserDesc(const std::string& name, std::vector<std::string>&& roles);

    UserDesc(const std::string& name, const std::string& description, std::vector<std::string>&& roles);

    /**
     * @brief Set the name of the user.
     * @param [in] name the name.
     */
    void
    SetName(const std::string& name);

    /**
     * @brief Get the name of the user.
     * @return the name.
     */
    const std::string&
    Name() const;

    /**
     * @brief Get the user description.
     * @return the description.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set the user description.
     *
     * @param [in] description
     */
    void
    SetDescription(const std::string& description);

    /**
     * @brief Add a role name for the user.
     * @param [in] role_name the role name.
     */
    void
    AddRole(const std::string& role_name);

    /**
     * @brief Get role names of the user.
     * @return the roles.
     */
    const std::vector<std::string>&
    Roles() const;

 private:
    std::string name_;
    std::string description_;
    std::vector<std::string> roles_;
};

}  // namespace milvus
