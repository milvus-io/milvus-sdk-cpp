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

#include "milvus/types/RoleDesc.h"

namespace milvus {

GrantItem::GrantItem(const std::string& object_type, const std::string& object_name, const std::string& db_name,
                     const std::string& role_name, const std::string& grantor_name, const std::string& privilege)
    : object_type_(object_type),
      object_name_(object_name),
      db_name_(db_name),
      role_name_(role_name),
      grantor_name_(grantor_name),
      privilege_(privilege) {
}

RoleDesc::RoleDesc() = default;

RoleDesc::RoleDesc(const std::string& name, std::vector<GrantItem>&& grant_items)
    : name_(name), grant_items_(std::move(grant_items)) {
}

void
RoleDesc::SetName(const std::string& name) {
    name_ = name;
}

const std::string&
RoleDesc::Name() const {
    return name_;
}

void
RoleDesc::AddGrantItem(GrantItem&& grant_item) {
    grant_items_.emplace_back(grant_item);
}

const std::vector<GrantItem>&
RoleDesc::GrantItems() const {
    return grant_items_;
}

}  // namespace milvus
