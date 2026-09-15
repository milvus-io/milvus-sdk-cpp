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
 * @brief Represnet a privilege item for role
 */
struct MILVUS_SDK_API GrantItem {
    /**
     * @brief Constructor
     */
    GrantItem(const std::string& object_type, const std::string& object_name, const std::string& db_name,
              const std::string& role_name, const std::string& grantor_name, const std::string& privilege);

    /**
     * @brief privilege type.
     */
    std::string object_type_;

    /**
     * @brief privilege name.
     */
    std::string object_name_;

    /**
     * @brief in which database take effect.
     */
    std::string db_name_;

    /**
     * @brief grant to which role.
     */
    std::string role_name_;

    /**
     * @brief privilege.
     */
    std::string privilege_;

    /**
     * @brief grantor name.
     */
    std::string grantor_name_;
};

/**
 * @brief Role description. Used by MilvusClient::DescribeRole().
 */
class MILVUS_SDK_API RoleDesc {
 public:
    /**
     * @brief Construct a new RoleDesc object.
     */
    RoleDesc();

    /**
     * @brief Construct a new RoleDesc object.
     * @param [in] name the name.
     * @param [in] grant_items the grant items.
     */
    RoleDesc(const std::string& name, std::vector<GrantItem>&& grant_items);

    RoleDesc(const std::string& name, const std::string& description, std::vector<GrantItem>&& grant_items);

    /**
     * @brief Set name of the role.
     * @param [in] name the name.
     */
    void
    SetName(const std::string& name);

    /**
     * @brief Get name of the role.
     * @return the name.
     */
    const std::string&
    Name() const;

    /**
     * @brief Get the role description.
     * @return the description.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set the role description.
     *
     * @param [in] description
     */
    void
    SetDescription(const std::string& description);

    /**
     * @brief Add a privilege item for the role.
     * @param [in] grant_item the grant item.
     */
    void
    AddGrantItem(GrantItem&& grant_item);

    /**
     * @brief Get privilege items of the role.
     * @return the grant items.
     */
    const std::vector<GrantItem>&
    GrantItems() const;

 private:
    std::string name_;
    std::string description_;
    std::vector<GrantItem> grant_items_;
};

}  // namespace milvus
