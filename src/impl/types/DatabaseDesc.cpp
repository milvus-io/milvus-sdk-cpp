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

#include "milvus/types/DatabaseDesc.h"

namespace milvus {

const std::string&
DatabaseDesc::Name() const {
    return db_name_;
}

void
DatabaseDesc::SetName(std::string name) {
    db_name_ = name;
}

int64_t
DatabaseDesc::ID() const {
    return db_id_;
}

void
DatabaseDesc::SetID(int64_t id) {
    db_id_ = id;
}

const std::unordered_map<std::string, std::string>&
DatabaseDesc::Properties() const {
    return properties_;
}

void
DatabaseDesc::SetProperties(const std::unordered_map<std::string, std::string>& properties) {
    properties_ = properties;
}

void
DatabaseDesc::SetProperties(std::unordered_map<std::string, std::string>&& properties) {
    properties_ = std::move(properties);
}

uint64_t
DatabaseDesc::CreatedTime() const {
    return created_timestamp_;
}

void
DatabaseDesc::SetCreatedTime(uint64_t ts) {
    created_timestamp_ = ts;
}

}  // namespace milvus
