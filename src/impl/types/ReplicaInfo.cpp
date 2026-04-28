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

#include "milvus/types/ReplicaInfo.h"

#include <utility>

namespace milvus {

int64_t
ReplicaInfo::ReplicaID() const {
    return replica_id_;
}

void
ReplicaInfo::SetReplicaID(int64_t replica_id) {
    replica_id_ = replica_id;
}

int64_t
ReplicaInfo::CollectionID() const {
    return collection_id_;
}

void
ReplicaInfo::SetCollectionID(int64_t collection_id) {
    collection_id_ = collection_id;
}

const std::vector<int64_t>&
ReplicaInfo::PartitionIDs() const {
    return partition_ids_;
}

void
ReplicaInfo::SetPartitionIDs(std::vector<int64_t>&& partition_ids) {
    partition_ids_ = std::move(partition_ids);
}

const std::vector<ShardReplica>&
ReplicaInfo::ShardReplicas() const {
    return shard_replicas_;
}

void
ReplicaInfo::SetShardReplicas(std::vector<ShardReplica>&& shard_replicas) {
    shard_replicas_ = std::move(shard_replicas);
}

const std::vector<int64_t>&
ReplicaInfo::NodeIDs() const {
    return node_ids_;
}

void
ReplicaInfo::SetNodeIDs(std::vector<int64_t>&& node_ids) {
    node_ids_ = std::move(node_ids);
}

const std::string&
ReplicaInfo::ResourceGroupName() const {
    return resource_group_name_;
}

void
ReplicaInfo::SetResourceGroupName(const std::string& resource_group_name) {
    resource_group_name_ = resource_group_name;
}

const std::unordered_map<std::string, int32_t>&
ReplicaInfo::NumOutboundNode() const {
    return num_outbound_node_;
}

void
ReplicaInfo::SetNumOutboundNode(std::unordered_map<std::string, int32_t>&& num_outbound_node) {
    num_outbound_node_ = std::move(num_outbound_node);
}

}  // namespace milvus
