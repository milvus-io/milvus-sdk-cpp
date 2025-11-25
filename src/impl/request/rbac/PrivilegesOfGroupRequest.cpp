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

#include "milvus/request/rbac/PrivilegesOfGroupRequest.h"

#include <memory>

namespace milvus {

const std::string&
PrivilegesOfGroupRequest::GroupName() const {
    return group_name_;
}

void
PrivilegesOfGroupRequest::SetGroupName(const std::string& name) {
    group_name_ = name;
}

PrivilegesOfGroupRequest&
PrivilegesOfGroupRequest::WithGroupName(const std::string& name) {
    group_name_ = name;
    return *this;
}

const std::set<std::string>&
PrivilegesOfGroupRequest::Privileges() const {
    return privileges_;
}

void
PrivilegesOfGroupRequest::SetPrivileges(std::set<std::string>&& privileges) {
    privileges_ = std::move(privileges);
}

PrivilegesOfGroupRequest&
PrivilegesOfGroupRequest::WithPrivileges(std::set<std::string>&& privileges) {
    privileges_ = std::move(privileges);
    return *this;
}

PrivilegesOfGroupRequest&
PrivilegesOfGroupRequest::AddPrivilege(const std::string& privilege) {
    privileges_.insert(privilege);
    return *this;
}

}  // namespace milvus
