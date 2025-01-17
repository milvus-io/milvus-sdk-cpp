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

DatabaseDesc::DatabaseDesc() = default;

DatabaseDesc::DatabaseDesc(const std::string& db_name, int64_t db_id, uint64_t created_timestamp,
                         const std::vector<std::pair<std::string, std::string>>& properties)
    : db_name_(db_name), db_id_(db_id), created_timestamp_(created_timestamp), properties_(properties) {}

const std::string&
DatabaseDesc::GetDbName() const {
    return db_name_;
}

int64_t
DatabaseDesc::GetDbID() const {
    return db_id_;
}

uint64_t
DatabaseDesc::GetCreatedTimestamp() const {
    return created_timestamp_;
}

const std::vector<std::pair<std::string, std::string>>&
DatabaseDesc::GetProperties() const {
    return properties_;
}

void
DatabaseDesc::SetDbName(const std::string& db_name) {
    db_name_ = db_name;
}

void
DatabaseDesc::SetDbID(int64_t db_id) {
    db_id_ = db_id;
}

void
DatabaseDesc::SetCreatedTimestamp(uint64_t created_timestamp) {
    created_timestamp_ = created_timestamp;
}

void
DatabaseDesc::SetProperties(const std::vector<std::pair<std::string, std::string>>& properties) {
    properties_ = properties;
}

void
DatabaseDesc::AddProperty(const std::string& key, const std::string& value) {
    properties_.emplace_back(key, value);
}

}  // namespace milvus
