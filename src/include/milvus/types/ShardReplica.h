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
#include <vector>

#include "milvus/Export.h"

namespace milvus {

class MILVUS_SDK_API ShardReplica {
 public:
    /**
     * @brief Constructor
     */
    ShardReplica() = default;

    /**
     * @brief Get leader node id.
     */
    int64_t
    LeaderID() const;

    /**
     * @brief Set leader node id.
     */
    void
    SetLeaderID(int64_t leader_id);

    /**
     * @brief Get leader address.
     */
    const std::string&
    LeaderAddress() const;

    /**
     * @brief Set leader address.
     */
    void
    SetLeaderAddress(const std::string& leader_address);

    /**
     * @brief Get channel name.
     */
    const std::string&
    ChannelName() const;

    /**
     * @brief Set channel name.
     */
    void
    SetChannelName(const std::string& channel_name);

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

 private:
    int64_t leader_id_{0};
    std::string leader_address_;
    std::string channel_name_;
    std::vector<int64_t> node_ids_;
};

}  // namespace milvus
