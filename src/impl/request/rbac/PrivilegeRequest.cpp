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

#include "milvus/request/rbac/PrivilegeRequest.h"

namespace milvus {

const std::string&
PrivilegeRequest::RoleName() const {
    return role_name_;
}

void
PrivilegeRequest::SetRoleName(const std::string& name) {
    role_name_ = name;
}

PrivilegeRequest&
PrivilegeRequest::WithRoleName(const std::string& name) {
    SetRoleName(name);
    return *this;
}

const std::string&
PrivilegeRequest::ObjectType() const {
    return object_type_;
}

void
PrivilegeRequest::SetObjectType(const std::string& object_type) {
    object_type_ = object_type;
}

PrivilegeRequest&
PrivilegeRequest::WithObjectType(const std::string& object_type) {
    SetObjectType(object_type);
    return *this;
}

const std::string&
PrivilegeRequest::ObjectName() const {
    return object_name_;
}

void
PrivilegeRequest::SetObjectName(const std::string& object_name) {
    object_name_ = object_name;
}

PrivilegeRequest&
PrivilegeRequest::WithObjectName(const std::string& object_name) {
    SetObjectName(object_name);
    return *this;
}

const std::string&
PrivilegeRequest::Privilege() const {
    return privilege_;
}

void
PrivilegeRequest::SetPrivilege(const std::string& privilege) {
    privilege_ = privilege;
}

PrivilegeRequest&
PrivilegeRequest::WithPrivilege(const std::string& privilege) {
    SetPrivilege(privilege);
    return *this;
}

const std::string&
PrivilegeRequest::DatabaseName() const {
    return db_name_;
}

void
PrivilegeRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

PrivilegeRequest&
PrivilegeRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

}  // namespace milvus
