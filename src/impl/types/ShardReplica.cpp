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

#include "milvus/types/ShardReplica.h"

#include <utility>

namespace milvus {

int64_t
ShardReplica::LeaderID() const {
    return leader_id_;
}

void
ShardReplica::SetLeaderID(int64_t leader_id) {
    leader_id_ = leader_id;
}

const std::string&
ShardReplica::LeaderAddress() const {
    return leader_address_;
}

void
ShardReplica::SetLeaderAddress(const std::string& leader_address) {
    leader_address_ = leader_address;
}

const std::string&
ShardReplica::ChannelName() const {
    return channel_name_;
}

void
ShardReplica::SetChannelName(const std::string& channel_name) {
    channel_name_ = channel_name;
}

const std::vector<int64_t>&
ShardReplica::NodeIDs() const {
    return node_ids_;
}

void
ShardReplica::SetNodeIDs(std::vector<int64_t>&& node_ids) {
    node_ids_ = std::move(node_ids);
}

}  // namespace milvus
