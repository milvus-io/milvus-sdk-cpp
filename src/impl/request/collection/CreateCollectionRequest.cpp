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

#include "milvus/request/collection/CreateCollectionRequest.h"

#include <memory>

namespace milvus {

const std::string&
CreateCollectionRequest::DatabaseName() const {
    return db_name_;
}

void
CreateCollectionRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

CreateCollectionRequest&
CreateCollectionRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const CollectionSchemaPtr&
CreateCollectionRequest::CollectionSchema() const {
    return schema_;
}

void
CreateCollectionRequest::SetCollectionSchema(const CollectionSchemaPtr& schema) {
    schema_ = schema;
}

CreateCollectionRequest&
CreateCollectionRequest::WithCollectionSchema(const CollectionSchemaPtr& schema) {
    SetCollectionSchema(schema);
    return *this;
}

int64_t
CreateCollectionRequest::NumPartitions() const {
    return num_partitions_;
}

void
CreateCollectionRequest::SetNumPartitions(int64_t num_partitions) {
    num_partitions_ = num_partitions;
}

CreateCollectionRequest&
CreateCollectionRequest::WithNumPartitions(int64_t num_partitions) {
    SetNumPartitions(num_partitions);
    return *this;
}

ConsistencyLevel
CreateCollectionRequest::GetConsistencyLevel() const {
    return level_;
}

void
CreateCollectionRequest::SetConsistencyLevel(ConsistencyLevel level) {
    level_ = level;
}

CreateCollectionRequest&
CreateCollectionRequest::WithConsistencyLevel(ConsistencyLevel level) {
    SetConsistencyLevel(level);
    return *this;
}

const std::unordered_map<std::string, std::string>&
CreateCollectionRequest::Properties() const {
    return properties_;
}

void
CreateCollectionRequest::SetProperties(std::unordered_map<std::string, std::string>&& properties) {
    properties_ = std::move(properties);
}

CreateCollectionRequest&
CreateCollectionRequest::WithProperties(std::unordered_map<std::string, std::string>&& properties) {
    SetProperties(std::move(properties));
    return *this;
}

CreateCollectionRequest&
CreateCollectionRequest::AddProperty(const std::string& key, const std::string& property) {
    properties_.insert(std::make_pair(key, property));
    return *this;
}

const std::vector<IndexDesc>&
CreateCollectionRequest::Indexes() const {
    return indexes_;
}

void
CreateCollectionRequest::SetIndexes(std::vector<IndexDesc>&& indexes) {
    indexes_ = std::move(indexes);
}

CreateCollectionRequest&
CreateCollectionRequest::WithIndexes(std::vector<IndexDesc>&& indexes) {
    SetIndexes(std::move(indexes));
    return *this;
}

CreateCollectionRequest&
CreateCollectionRequest::AddIndex(IndexDesc&& index) {
    indexes_.emplace_back(std::move(index));
    return *this;
}

}  // namespace milvus
