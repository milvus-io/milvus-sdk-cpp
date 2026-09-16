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
 * @brief Used by MilvusClientV2::CreateRole().
 */
class MILVUS_SDK_API CreateRoleRequest {
 public:
    /**
     * @brief Constructor
     */
    CreateRoleRequest() = default;

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
    CreateRoleRequest&
    WithRoleName(const std::string& name);

    /**
     * @brief Description of the role.
     * @return the description.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set description of the role.
     * @param [in] description the description.
     */
    void
    SetDescription(const std::string& description);

    /**
     * @brief Set description of the role.
     * @param [in] description the description.
     */
    CreateRoleRequest&
    WithDescription(const std::string& description);

 protected:
    std::string role_name_;
    std::string description_;
};

}  // namespace milvus
