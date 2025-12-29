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
#include <string>
#include <unordered_map>
#include <vector>

#include "ResourceGroupConfig.h"

namespace milvus {

/**
 * @brief NodeInfo is used to describe a node information.
 */
struct NodeInfo {
    /**
     * @brief Constructor
     */
    NodeInfo(int64_t id, const std::string& address, const std::string& hostname);

    /**
     * @brief id of this node.
     */
    int64_t id_{0};

    /**
     * @brief address of this node.
     */
    std::string address_;

    /**
     * @brief hostname of this node.
     */
    std::string hostname_;
};

/**
 * @brief Resource group descriptions for MilvusClient::CreateResourceGroup()
 */
class ResourceGroupDesc {
 public:
    ResourceGroupDesc();

    /**
     * @brief Name of the resource group.
     */
    const std::string&
    Name() const;

    /**
     * @brief Set name of the resource group.
     */
    void
    SetName(const std::string& name);

    /**
     * @brief Capacity of requested nodes.
     */
    uint32_t
    Capacity() const;

    /**
     * @brief Set capacity of requested nodes.
     */
    void
    SetCapacity(uint32_t capacity);

    /**
     * @brief Number of available nodes.
     */
    uint32_t
    AvailableNodesNum() const;

    /**
     * @brief Set number of available nodes.
     */
    void
    SetAvailableNodesNum(uint32_t num);

    /**
     * @brief Loaded replica number for each collection.
     */
    const std::unordered_map<std::string, uint32_t>&
    LoadedReplicasNum() const;

    /**
     * @brief Set loaded replica number for a collection.
     */
    void
    AddLoadedReplicasNum(const std::string& collection_name, uint32_t num);

    /**
     * @brief Accessed other resource group's nodes number for each collection.
     */
    const std::unordered_map<std::string, uint32_t>&
    OutgoingNodesNum() const;

    /**
     * @brief Set accessed other resource group's nodes number of a collection.
     */
    void
    AddOutgoingNodesNum(const std::string& collection_name, uint32_t num);

    /**
     * @brief The number of nodes be accessed by other resource groups for each collection.
     */
    const std::unordered_map<std::string, uint32_t>&
    IncomingNodesNum() const;

    /**
     * @brief Set the number of nodes be accessed by other resource groups of a collection.
     */
    void
    AddIncomingNodesNum(const std::string& collection_name, uint32_t num);

    /**
     * @brief Configurations of the resource group.
     */
    const ResourceGroupConfig&
    Config() const;

    /**
     * @brief Set configurations of the resource group.
     */
    void
    SetConfig(ResourceGroupConfig&& config);

    /**
     * @brief Basic informations of all nodes in the resource group.
     */
    const std::vector<NodeInfo>&
    Nodes() const;

    /**
     * @brief Add a basic informations of a node.
     */
    void
    AddNode(NodeInfo&& node);

 private:
    std::string name_;
    uint32_t capacity_{0};
    uint32_t num_available_node_{0};
    std::unordered_map<std::string, uint32_t> num_loaded_replica_;
    std::unordered_map<std::string, uint32_t> num_outgoing_node_;
    std::unordered_map<std::string, uint32_t> num_incoming_node_;

    ResourceGroupConfig config_;
    std::vector<NodeInfo> nodes_;
};

}  // namespace milvus
