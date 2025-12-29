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
#include <unordered_map>

#include "milvus/types/ResourceGroupConfig.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::UpdateResourceGroups()
 */
class UpdateResourceGroupsRequest {
 public:
    /**
     * @brief Constructor
     */
    UpdateResourceGroupsRequest() = default;

    /**
     * @brief Get the resource groups to be updated.
     */
    const std::unordered_map<std::string, ResourceGroupConfig>&
    Groups() const;

    /**
     * @brief Set the resource groups to be updated.
     */
    void
    SetGroups(std::unordered_map<std::string, ResourceGroupConfig>&& groups);

    /**
     * @brief Set the resource groups to be updated.
     */
    UpdateResourceGroupsRequest&
    WithGroups(std::unordered_map<std::string, ResourceGroupConfig>&& groups);

    /**
     * @brief Add a resource group to be updated.
     */
    UpdateResourceGroupsRequest&
    AddGroup(const std::string& name, ResourceGroupConfig&& config);

 private:
    std::unordered_map<std::string, ResourceGroupConfig> groups_;
};

}  // namespace milvus
