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

NodeInfo::NodeInfo(int64_t id, const std::string& address, const std::string& hostname)
    : id_(id), address_(address), hostname_(hostname) {
}

ResourceGroupDesc::ResourceGroupDesc() = default;

const std::string&
ResourceGroupDesc::Name() const {
    return name_;
}

void
ResourceGroupDesc::SetName(const std::string& name) {
    name_ = name;
}

uint32_t
ResourceGroupDesc::ResourceGroupDesc::Capacity() const {
    return capacity_;
}

void
ResourceGroupDesc::SetCapacity(uint32_t capacity) {
    capacity_ = capacity;
}

uint32_t
ResourceGroupDesc::AvailableNodesNum() const {
    return num_available_node_;
}

void
ResourceGroupDesc::SetAvailableNodesNum(uint32_t num) {
    num_available_node_ = num;
}

const std::unordered_map<std::string, uint32_t>&
ResourceGroupDesc::LoadedReplicasNum() const {
    return num_loaded_replica_;
}

void
ResourceGroupDesc::AddLoadedReplicasNum(const std::string& collection_name, uint32_t num) {
    num_loaded_replica_.insert(std::make_pair(collection_name, num));
}

const std::unordered_map<std::string, uint32_t>&
ResourceGroupDesc::OutgoingNodesNum() const {
    return num_outgoing_node_;
}

void
ResourceGroupDesc::AddOutgoingNodesNum(const std::string& collection_name, uint32_t num) {
    num_outgoing_node_.insert(std::make_pair(collection_name, num));
}

const std::unordered_map<std::string, uint32_t>&
ResourceGroupDesc::IncomingNodesNum() const {
    return num_incoming_node_;
}

void
ResourceGroupDesc::AddIncomingNodesNum(const std::string& collection_name, uint32_t num) {
    num_incoming_node_.insert(std::make_pair(collection_name, num));
}

const ResourceGroupConfig&
ResourceGroupDesc::Config() const {
    return config_;
}

void
ResourceGroupDesc::SetConfig(ResourceGroupConfig&& config) {
    config_ = std::move(config);
}

const std::vector<NodeInfo>&
ResourceGroupDesc::Nodes() const {
    return nodes_;
}

void
ResourceGroupDesc::AddNode(NodeInfo&& node) {
    nodes_.emplace_back(std::move(node));
}

}  // namespace milvus
