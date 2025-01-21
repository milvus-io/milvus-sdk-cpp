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

ResourceGroupConfig::ResourceGroupConfig(int req_node_num, int lim_node_num, const std::vector<std::string>& from,
                                         const std::vector<std::string>& to,
                                         const std::vector<std::pair<std::string, std::string>>& labels)
    : requests_node_num_(req_node_num),
      limits_node_num_(lim_node_num),
      transfer_from_(from),
      transfer_to_(to),
      node_labels_(labels) {
}

int
ResourceGroupConfig::GetRequestsNodeNum() const {
    return requests_node_num_;
}

void
ResourceGroupConfig::SetRequestsNodeNum(int num) {
    requests_node_num_ = num;
}

int
ResourceGroupConfig::GetLimitsNodeNum() const {
    return limits_node_num_;
}

void
ResourceGroupConfig::SetLimitsNodeNum(int num) {
    limits_node_num_ = num;
}

const std::vector<std::string>&
ResourceGroupConfig::GetTransferFrom() const {
    return transfer_from_;
}

void
ResourceGroupConfig::SetTransferFrom(const std::vector<std::string>& from) {
    transfer_from_ = from;
}

const std::vector<std::string>&
ResourceGroupConfig::GetTransferTo() const {
    return transfer_to_;
}

void
ResourceGroupConfig::SetTransferTo(const std::vector<std::string>& to) {
    transfer_to_ = to;
}

const std::vector<std::pair<std::string, std::string>>&
ResourceGroupConfig::GetNodeLabels() const {
    return node_labels_;
}

void
ResourceGroupConfig::SetNodeLabels(const std::vector<std::pair<std::string, std::string>>& labels) {
    node_labels_ = labels;
}

}  // namespace milvus
