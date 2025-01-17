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

#include "milvus/types/PrivilegeGroupInfo.h"

PrivilegeGroupInfo::PrivilegeGroupInfo(const std::string& group_name) : group_name(group_name) {}

void PrivilegeGroupInfo::AddPrivilege(const std::string& privilege) {
    privileges.push_back(privilege);
}

const std::string& PrivilegeGroupInfo::GroupName() const {
    return group_name;
}

void PrivilegeGroupInfo::SetGroupName(const std::string& group_name) {
    this->group_name = group_name;
}

const std::vector<std::string>& PrivilegeGroupInfo::Privileges() const {
    return privileges;
}

void PrivilegeGroupInfo::SetPrivileges(const std::vector<std::string>& privileges) {
    this->privileges = privileges;
}
