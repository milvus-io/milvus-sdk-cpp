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

namespace milvus {

/**
 * @brief Used by MilvusClientV2::GrantPrivilegeV2() and RevokePrivilegeV2().
 */
class PrivilegeV2Request {
 public:
    /**
     * @brief Constructor
     */
    PrivilegeV2Request() = default;

    /**
     * @brief Name of the role.
     */
    const std::string&
    RoleName() const;

    /**
     * @brief Set name of the role.
     */
    void
    SetRoleName(const std::string& name);

    /**
     * @brief Set name of the role.
     */
    PrivilegeV2Request&
    WithRoleName(const std::string& name);

    /**
     * @brief Get database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name.
     */
    PrivilegeV2Request&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Name of the collection.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of the collection.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set name of the collection.
     */
    PrivilegeV2Request&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Name of the privilege.
     */
    const std::string&
    Privilege() const;

    /**
     * @brief Set name of the privilege.
     */
    void
    SetPrivilege(const std::string& privilege);

    /**
     * @brief Set name of the privilege.
     */
    PrivilegeV2Request&
    WithPrivilege(const std::string& privilege);

 protected:
    std::string role_name_;
    std::string db_name_;
    std::string collection_name_;
    std::string privilege_;
};

using GrantPrivilegeV2Request = PrivilegeV2Request;
using RevokePrivilegeV2Request = PrivilegeV2Request;

}  // namespace milvus
