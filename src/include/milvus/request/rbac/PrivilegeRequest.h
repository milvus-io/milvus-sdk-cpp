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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::GrantPrivilege() and RevokePrivilege().
 *
 * The object-scoped (V1) privilege form grants a privilege on an arbitrary object,
 * identified by its type and name. Typical object types are "Global", "Database",
 * "Collection" and "User"; the object name is empty for the "Global" scope.
 */
class MILVUS_SDK_API PrivilegeRequest {
 public:
    /**
     * @brief Constructor
     */
    PrivilegeRequest() = default;

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
    PrivilegeRequest&
    WithRoleName(const std::string& name);

    /**
     * @brief Type of the object the privilege applies to, such as "Global", "Database",
     * "Collection" or "User".
     */
    const std::string&
    ObjectType() const;

    /**
     * @brief Set type of the object the privilege applies to.
     */
    void
    SetObjectType(const std::string& object_type);

    /**
     * @brief Set type of the object the privilege applies to.
     */
    PrivilegeRequest&
    WithObjectType(const std::string& object_type);

    /**
     * @brief Name of the object the privilege applies to. Empty for the "Global" scope.
     */
    const std::string&
    ObjectName() const;

    /**
     * @brief Set name of the object the privilege applies to.
     */
    void
    SetObjectName(const std::string& object_name);

    /**
     * @brief Set name of the object the privilege applies to.
     */
    PrivilegeRequest&
    WithObjectName(const std::string& object_name);

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
    PrivilegeRequest&
    WithPrivilege(const std::string& privilege);

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
    PrivilegeRequest&
    WithDatabaseName(const std::string& db_name);

 protected:
    std::string role_name_;
    std::string object_type_;
    std::string object_name_;
    std::string privilege_;
    std::string db_name_;
};

using GrantPrivilegeRequest = PrivilegeRequest;
using RevokePrivilegeRequest = PrivilegeRequest;

}  // namespace milvus
