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

#include "milvus/request/resourcegroup/UpdateResourceGroupsRequest.h"

#include <memory>

namespace milvus {

const std::unordered_map<std::string, ResourceGroupConfig>&
UpdateResourceGroupsRequest::Groups() const {
    return groups_;
}

void
UpdateResourceGroupsRequest::SetGroups(std::unordered_map<std::string, ResourceGroupConfig>&& groups) {
    groups_ = std::move(groups);
}

UpdateResourceGroupsRequest&
UpdateResourceGroupsRequest::WithGroups(std::unordered_map<std::string, ResourceGroupConfig>&& groups) {
    SetGroups(std::move(groups));
    return *this;
}

UpdateResourceGroupsRequest&
UpdateResourceGroupsRequest::AddGroup(const std::string& name, ResourceGroupConfig&& config) {
    groups_.emplace(name, std::move(config));
    return *this;
}

}  // namespace milvus
