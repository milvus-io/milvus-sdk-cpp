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

#include "milvus/request/dml/DeleteRequest.h"

#include <memory>

namespace milvus {

const std::string&
DeleteRequest::DatabaseName() const {
    return db_name_;
}

void
DeleteRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

DeleteRequest&
DeleteRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::string&
DeleteRequest::CollectionName() const {
    return collection_name_;
}

void
DeleteRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

DeleteRequest&
DeleteRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

const std::string&
DeleteRequest::PartitionName() const {
    return partition_name_;
}

void
DeleteRequest::SetPartitionName(const std::string& partition_name) {
    partition_name_ = partition_name;
}

DeleteRequest&
DeleteRequest::WithPartitionName(const std::string& partition_name) {
    SetPartitionName(partition_name);
    return *this;
}

const std::string&
DeleteRequest::Filter() const {
    return filter_;
}

void
DeleteRequest::SetFilter(const std::string& filter) {
    filter_ = filter;
}

DeleteRequest&
DeleteRequest::WithFilter(const std::string& filter) {
    SetFilter(filter);
    return *this;
}

const std::unordered_map<std::string, nlohmann::json>&
DeleteRequest::FilterTemplates() const {
    return filter_templates_;
}

DeleteRequest&
DeleteRequest::AddFilterTemplate(std::string key, nlohmann::json&& filter_template) {
    filter_templates_[std::move(key)] = std::move(filter_template);
    return *this;
}

void
DeleteRequest::SetFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates) {
    filter_templates_ = std::move(filter_templates);
}

DeleteRequest&
DeleteRequest::WithFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates) {
    SetFilterTemplates(std::move(filter_templates));
    return *this;
}

DeleteRequest&
DeleteRequest::AddFilterTemplate(const std::string& key, nlohmann::json&& filter_template) {
    filter_templates_[key] = std::move(filter_template);
    return *this;
}

const IDArray&
DeleteRequest::IDs() const {
    return ids_;
}

void
DeleteRequest::SetIDs(std::vector<int64_t>&& id_array) {
    ids_ = IDArray(std::move(id_array));
}

DeleteRequest&
DeleteRequest::WithIDs(std::vector<int64_t>&& id_array) {
    SetIDs(std::move(id_array));
    return *this;
}

void
DeleteRequest::SetIDs(std::vector<std::string>&& id_array) {
    ids_ = IDArray(std::move(id_array));
}

DeleteRequest&
DeleteRequest::WithIDs(std::vector<std::string>&& id_array) {
    SetIDs(std::move(id_array));
    return *this;
}

}  // namespace milvus
