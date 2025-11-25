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

#include <set>
#include <string>

namespace milvus {

/**
 * @brief Used by MilvusClientV2::AddPrivilegesToGroup() and RemovePrivilegesFromGroup().
 */
class PrivilegesOfGroupRequest {
 public:
    /**
     * @brief Constructor
     */
    PrivilegesOfGroupRequest() = default;

    /**
     * @brief Name of the privilege group.
     */
    const std::string&
    GroupName() const;

    /**
     * @brief Set name of the privilege group..
     */
    void
    SetGroupName(const std::string& name);

    /**
     * @brief Set name of the privilege group..
     */
    PrivilegesOfGroupRequest&
    WithGroupName(const std::string& name);

    /**
     * @brief Get privileges of the group to be added or removed.
     */
    const std::set<std::string>&
    Privileges() const;

    /**
     * @brief Set privileges of the group to be added or removed.
     */
    void
    SetPrivileges(std::set<std::string>&& privileges);

    /**
     * @brief Set privileges of the group to be added or removed.
     */
    PrivilegesOfGroupRequest&
    WithPrivileges(std::set<std::string>&& privileges);

    /**
     * @brief Add a privileges of the group to be added or removed.
     */
    PrivilegesOfGroupRequest&
    AddPrivilege(const std::string& privilege);

 protected:
    std::string group_name_;
    std::set<std::string> privileges_;
};

using AddPrivilegesToGroupRequest = PrivilegesOfGroupRequest;
using RemovePrivilegesFromGroupRequest = PrivilegesOfGroupRequest;

}  // namespace milvus
