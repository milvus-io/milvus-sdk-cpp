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

#include <map>
#include <string>
#include <vector>

#include "NodeInfo.h"
#include "ResourceGroupConfig.h"

namespace milvus {

class ResourceGroupDesc {
 public:
    ResourceGroupDesc() = default;
    ResourceGroupDesc(const std::string& name, int32_t capacity, int32_t available_nodes,
                      const std::map<std::string, int32_t>& loaded_replicas,
                      const std::map<std::string, int32_t>& outgoing_nodes,
                      const std::map<std::string, int32_t>& incoming_nodes, const ResourceGroupConfig& config,
                      const std::vector<NodeInfo>& nodes);

    const std::string&
    GetName() const;
    int32_t
    GetCapacity() const;
    int32_t
    GetNumAvailableNode() const;
    const std::map<std::string, int32_t>&
    GetNumLoadedReplica() const;
    const std::map<std::string, int32_t>&
    GetNumOutgoingNode() const;
    const std::map<std::string, int32_t>&
    GetNumIncomingNode() const;
    const ResourceGroupConfig&
    GetConfig() const;
    const std::vector<NodeInfo>&
    GetNodes() const;

 private:
    std::string name_;
    int32_t capacity_;
    int32_t num_available_node_;
    std::map<std::string, int32_t> num_loaded_replica_;
    std::map<std::string, int32_t> num_outgoing_node_;
    std::map<std::string, int32_t> num_incoming_node_;
    ResourceGroupConfig config_;
    std::vector<NodeInfo> nodes_;
};

}  // namespace milvus
