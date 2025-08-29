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
 * @brief Represnet a privilege item for role
 */
struct GrantItem {
    /**
     * @brief Constructor
     */
    GrantItem(const std::string& object_type, const std::string& object_name, const std::string& db_name,
              const std::string& role_name, const std::string& grantor_name, const std::string& privilege);

    /**
     * @brief privilege type
     */
    std::string object_type_;

    /**
     * @brief privilege name
     */
    std::string object_name_;

    /**
     * @brief in which database take effect
     */
    std::string db_name_;

    /**
     * @brief grant to which role
     */
    std::string role_name_;

    /**
     * @brief privilege
     */
    std::string privilege_;

    /**
     * @brief grantor name
     */
    std::string grantor_name_;
};

/**
 * @brief Role description. Used by MilvusClient::DescribeRole().
 */
class RoleDesc {
 public:
    /**
     * @brief Construct a new RoleDesc object
     */
    RoleDesc();

    /**
     * @brief Construct a new RoleDesc object
     */
    RoleDesc(const std::string& name, std::vector<GrantItem>&& grant_items);

    void
    SetName(const std::string& name);

    const std::string&
    Name() const;

    void
    AddGrantItem(GrantItem&& grant_item);

    const std::vector<GrantItem>&
    GrantItems() const;

 private:
    std::string name_;
    std::vector<GrantItem> grant_items_;
};

}  // namespace milvus
