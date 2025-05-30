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

#include <string>
#include <utility>
#include <vector>

namespace milvus {

class ResourceGroupConfig {
 public:
    ResourceGroupConfig() = default;
    ResourceGroupConfig(int req_node_num, int lim_node_num, const std::vector<std::string>& from,
                        const std::vector<std::string>& to,
                        const std::vector<std::pair<std::string, std::string>>& labels);

    int
    GetRequestsNodeNum() const;
    void
    SetRequestsNodeNum(int num);

    int
    GetLimitsNodeNum() const;
    void
    SetLimitsNodeNum(int num);

    const std::vector<std::string>&
    GetTransferFrom() const;
    void
    SetTransferFrom(const std::vector<std::string>& from);

    const std::vector<std::string>&
    GetTransferTo() const;
    void
    SetTransferTo(const std::vector<std::string>& to);

    const std::vector<std::pair<std::string, std::string>>&
    GetNodeLabels() const;
    void
    SetNodeLabels(const std::vector<std::pair<std::string, std::string>>& labels);

 private:
    int requests_node_num_;
    int limits_node_num_;
    std::vector<std::string> transfer_from_;
    std::vector<std::string> transfer_to_;
    std::vector<std::pair<std::string, std::string>> node_labels_;
};

}  // namespace milvus
