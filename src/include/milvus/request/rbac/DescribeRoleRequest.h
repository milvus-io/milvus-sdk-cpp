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
 * @brief Used by MilvusClientV2::DescribeRole().
 */
class MILVUS_SDK_API DescribeRoleRequest {
 public:
    /**
     * @brief Constructor
     */
    DescribeRoleRequest() = default;

    /**
     * @brief Name of the role.
     * @return the role name.
     */
    const std::string&
    RoleName() const;

    /**
     * @brief Set name of the role.
     * @param [in] name the name.
     */
    void
    SetRoleName(const std::string& name);

    /**
     * @brief Set name of the role.
     * @param [in] name the name.
     */
    DescribeRoleRequest&
    WithRoleName(const std::string& name);

    /**
     * @brief Database name which the role is assigned.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name which the role is assigned.
     * @param [in] db_name the DB name.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name which the role is assigned.
     * @param [in] db_name the DB name.
     */
    DescribeRoleRequest&
    WithDatabaseName(const std::string& db_name);

 protected:
    std::string role_name_;
    std::string db_name_;
};

}  // namespace milvus
