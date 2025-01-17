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

#include "milvus/types/ResourceGroupDesc.h"

namespace milvus {

ResourceGroupDesc::ResourceGroupDesc(const std::string& name, int32_t capacity, int32_t available_nodes,
                                     const std::map<std::string, int32_t>& loaded_replicas,
                                     const std::map<std::string, int32_t>& outgoing_nodes,
                                     const std::map<std::string, int32_t>& incoming_nodes,
                                     const ResourceGroupConfig& config,
                                     const std::vector<NodeInfo>& nodes)
    : name(name), capacity(capacity), num_available_node(available_nodes),
      num_loaded_replica(loaded_replicas), num_outgoing_node(outgoing_nodes),
      num_incoming_node(incoming_nodes), config(config), nodes(nodes) {}

const std::string& ResourceGroupDesc::GetName() const {
    return name;
}

int32_t ResourceGroupDesc::GetCapacity() const {
    return capacity;
}

int32_t ResourceGroupDesc::GetNumAvailableNode() const {
    return num_available_node;
}

const std::map<std::string, int32_t>& ResourceGroupDesc::GetNumLoadedReplica() const {
    return num_loaded_replica;
}

const std::map<std::string, int32_t>& ResourceGroupDesc::GetNumOutgoingNode() const {
    return num_outgoing_node;
}

const std::map<std::string, int32_t>& ResourceGroupDesc::GetNumIncomingNode() const {
    return num_incoming_node;
}

const ResourceGroupConfig& ResourceGroupDesc::GetConfig() const {
    return config;
}

const std::vector<NodeInfo>& ResourceGroupDesc::GetNodes() const {
    return nodes;
}

}  // namespace milvus
