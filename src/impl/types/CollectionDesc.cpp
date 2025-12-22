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

#include "milvus/types/CollectionDesc.h"

namespace milvus {

const std::string&
CollectionDesc::DatabaseName() const {
    return db_name_;
}

void
CollectionDesc::SetDatabaseName(std::string name) {
    db_name_ = name;
}

const CollectionSchema&
CollectionDesc::Schema() const {
    return schema_;
}

void
CollectionDesc::SetSchema(const CollectionSchema& schema) {
    schema_ = schema;
}

void
CollectionDesc::SetSchema(CollectionSchema&& schema) {
    schema_ = std::move(schema);
}

int64_t
CollectionDesc::ID() const {
    return collection_id_;
}

void
CollectionDesc::SetID(int64_t id) {
    collection_id_ = id;
}

const std::vector<std::string>&
CollectionDesc::Alias() const {
    return alias_;
}

void
CollectionDesc::SetAlias(const std::vector<std::string>& alias) {
    alias_ = alias;
}

void
CollectionDesc::SetAlias(std::vector<std::string>&& alias) {
    alias_ = std::move(alias);
}

uint64_t
CollectionDesc::CreatedTime() const {
    return created_utc_timestamp_;
}

void
CollectionDesc::SetCreatedTime(uint64_t ts) {
    created_utc_timestamp_ = ts;
}

uint64_t
CollectionDesc::UpdateTime() const {
    return update_timestamp_;
}

void
CollectionDesc::SetUpdateTime(uint64_t ts) {
    update_timestamp_ = ts;
}

const std::unordered_map<std::string, std::string>&
CollectionDesc::Properties() const {
    return properties_;
}

void
CollectionDesc::SetProperties(std::unordered_map<std::string, std::string>&& properties) {
    properties_ = std::move(properties);
}

}  // namespace milvus
