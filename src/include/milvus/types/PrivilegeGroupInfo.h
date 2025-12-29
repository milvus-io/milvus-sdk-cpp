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
 * @brief Privilege group information. Used by MilvusClient::ListPrivilegeGroups().
 */
class PrivilegeGroupInfo {
 public:
    /**
     * @brief Construct a new PrivilegeGroupInfo object.
     */
    PrivilegeGroupInfo();

    /**
     * @brief Construct a new PrivilegeGroupInfo object.
     */
    PrivilegeGroupInfo(const std::string& name, std::vector<std::string>&& privileges);

    /**
     * @brief Name of the group.
     */
    const std::string&
    Name() const;

    /**
     * @brief Set name of the group.
     */
    void
    SetName(const std::string& name);

    /**
     * @brief Privileges if the group.
     */
    const std::vector<std::string>&
    Privileges() const;

    /**
     * @brief Add a privilege name into the info.
     */
    void
    AddPrivilege(const std::string& privilege);

 private:
    std::string name_;
    std::vector<std::string> privileges_;
};

/**
 * @brief PrivilegeGroupInfo objects array
 */
using PrivilegeGroupInfos = std::vector<PrivilegeGroupInfo>;

}  // namespace milvus
