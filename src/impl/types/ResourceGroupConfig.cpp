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

#include "milvus/types/ResourceGroupConfig.h"

namespace milvus {

ResourceGroupConfig::ResourceGroupConfig() = default;

uint32_t
ResourceGroupConfig::Requests() const {
    return requests_;
}

void
ResourceGroupConfig::SetRequests(uint32_t num) {
    requests_ = num;
}

uint32_t
ResourceGroupConfig::Limits() const {
    return limits_;
}

void
ResourceGroupConfig::SetLimits(uint32_t num) {
    limits_ = num;
}

const std::set<std::string>&
ResourceGroupConfig::TransferFromGroups() const {
    return transfer_from_;
}

void
ResourceGroupConfig::AddTrnasferFromGroup(const std::string& group_name) {
    transfer_from_.insert(group_name);
}

const std::set<std::string>&
ResourceGroupConfig::TransferToGroups() const {
    return transfer_to_;
}

void
ResourceGroupConfig::AddTrnasferToGroup(const std::string& group_name) {
    transfer_to_.insert(group_name);
}

const std::unordered_map<std::string, std::string>&
ResourceGroupConfig::NodeFilters() const {
    return node_filters_;
}

void
ResourceGroupConfig::AddNodeFilter(const std::string& key, const std::string& value) {
    node_filters_.insert(std::make_pair(key, value));
}

}  // namespace milvus
