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

#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>

namespace milvus {

/**
 * @brief Resource group descriptions for MilvusClient::CreateResourceGroup()
 */
class ResourceGroupConfig {
 public:
    ResourceGroupConfig();

    /**
     * @brief Number of requested nodes.
     */
    uint32_t
    Requests() const;

    /**
     * @brief Set number of requested nodes.
     */
    void
    SetRequests(uint32_t num);

    /**
     * @brief Maximum number of nodes.
     */
    uint32_t
    Limits() const;

    /**
     * @brief Set maximum number of nodes.
     */
    void
    SetLimits(uint32_t num);

    /**
     * @brief Group names from which the nodes can be transfered.
     */
    const std::set<std::string>&
    TransferFromGroups() const;

    /**
     * @brief Add a group name from which the nodes can be transfered.
     */
    void
    AddTrnasferFromGroup(const std::string& group_name);

    /**
     * @brief Group names to which the nodes can be transfered.
     */
    const std::set<std::string>&
    TransferToGroups() const;

    /**
     * @brief Add a group name to which the nodes can be transfered.
     */
    void
    AddTrnasferToGroup(const std::string& group_name);

    /**
     * @brief Resource group will prefer to accept node which match node filter.
     */
    const std::unordered_map<std::string, std::string>&
    NodeFilters() const;

    /**
     * @brief Add a node filter, each filter is a key-value pair that represent a label of nodes.
     * For example, a node is marked as "CPU : 32", the key is "CPU", value is "32".
     */
    void
    AddNodeFilter(const std::string& key, const std::string& value);

 private:
    uint32_t requests_{0};
    uint32_t limits_{0};
    std::set<std::string> transfer_from_;
    std::set<std::string> transfer_to_;
    std::unordered_map<std::string, std::string> node_filters_;
};

}  // namespace milvus
