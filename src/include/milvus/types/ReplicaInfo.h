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

#include "ShardReplica.h"

namespace milvus {

class ReplicaInfo {
 public:
    /**
     * @brief Constructor
     */
    ReplicaInfo() = default;

    /**
     * @brief Get replica id.
     */
    int64_t
    ReplicaID() const;

    /**
     * @brief Set replica id.
     */
    void
    SetReplicaID(int64_t replica_id);

    /**
     * @brief Get collection id.
     */
    int64_t
    CollectionID() const;

    /**
     * @brief Set collection id.
     */
    void
    SetCollectionID(int64_t collection_id);

    /**
     * @brief Get partition ids.
     */
    const std::vector<int64_t>&
    PartitionIDs() const;

    /**
     * @brief Set partition ids.
     */
    void
    SetPartitionIDs(std::vector<int64_t>&& partition_ids);

    /**
     * @brief Get shard replicas.
     */
    const std::vector<ShardReplica>&
    ShardReplicas() const;

    /**
     * @brief Set shard replicas.
     */
    void
    SetShardReplicas(std::vector<ShardReplica>&& shard_replicas);

    /**
     * @brief Get node ids.
     */
    const std::vector<int64_t>&
    NodeIDs() const;

    /**
     * @brief Set node ids.
     */
    void
    SetNodeIDs(std::vector<int64_t>&& node_ids);

    /**
     * @brief Get resource group name.
     */
    const std::string&
    ResourceGroupName() const;

    /**
     * @brief Set resource group name.
     */
    void
    SetResourceGroupName(const std::string& resource_group_name);

    /**
     * @brief Get outbound node count by resource group.
     */
    const std::unordered_map<std::string, int32_t>&
    NumOutboundNode() const;

    /**
     * @brief Set outbound node count by resource group.
     */
    void
    SetNumOutboundNode(std::unordered_map<std::string, int32_t>&& num_outbound_node);

 private:
    int64_t replica_id_{0};
    int64_t collection_id_{0};
    std::vector<int64_t> partition_ids_;
    std::vector<ShardReplica> shard_replicas_;
    std::vector<int64_t> node_ids_;
    std::string resource_group_name_;
    std::unordered_map<std::string, int32_t> num_outbound_node_;
};

}  // namespace milvus
