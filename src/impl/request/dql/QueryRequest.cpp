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

#include "milvus/request/dql/QueryRequest.h"

#include <memory>

#include "../../utils/ExtraParamUtils.h"

namespace milvus {

QueryRequest&
QueryRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

QueryRequest&
QueryRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

QueryRequest&
QueryRequest::WithPartitionNames(std::set<std::string>&& partition_names) {
    SetPartitionNames(std::move(partition_names));
    return *this;
}

QueryRequest&
QueryRequest::AddPartitionName(const std::string& partition_name) {
    DQLRequestBase::AddPartitionName(partition_name);
    return *this;
}

QueryRequest&
QueryRequest::WithOutputFields(std::set<std::string>&& output_field_names) {
    SetOutputFields(std::move(output_field_names));
    return *this;
}

QueryRequest&
QueryRequest::AddOutputField(const std::string& output_field) {
    DQLRequestBase::AddOutputField(output_field);
    return *this;
}

QueryRequest&
QueryRequest::WithConsistencyLevel(ConsistencyLevel consistency_level) {
    SetConsistencyLevel(consistency_level);
    return *this;
}

const std::string&
QueryRequest::Filter() const {
    return filter_;
}

void
QueryRequest::SetFilter(std::string filter) {
    filter_ = std::move(filter);
}

QueryRequest&
QueryRequest::WithFilter(std::string filter) {
    SetFilter(std::move(filter));
    return *this;
}

const std::unordered_map<std::string, nlohmann::json>&
QueryRequest::FilterTemplates() const {
    return filter_templates_;
}

void
QueryRequest::SetFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates) {
    filter_templates_ = std::move(filter_templates);
}

QueryRequest&
QueryRequest::AddFilterTemplate(std::string key, const nlohmann::json& filter_template) {
    filter_templates_.emplace(std::move(key), filter_template);
    return *this;
}

QueryRequest&
QueryRequest::WithFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates) {
    SetFilterTemplates(std::move(filter_templates));
    return *this;
}

int64_t
QueryRequest::Limit() const {
    // for history reason, query() requires "limit", search() requires "topk"
    return GetExtraInt64(extra_params_, "limit", 0);
}

void
QueryRequest::SetLimit(int64_t limit) {
    // for history reason, query() requires "limit", search() requires "topk"
    SetExtraInt64(extra_params_, "limit", limit);
}

QueryRequest&
QueryRequest::WithLimit(int64_t limit) {
    SetLimit(limit);
    return *this;
}

int64_t
QueryRequest::Offset() const {
    return GetExtraInt64(extra_params_, "offset", 0);
}

void
QueryRequest::SetOffset(int64_t offset) {
    SetExtraInt64(extra_params_, "offset", offset);
}

QueryRequest&
QueryRequest::WithOffset(int64_t offset) {
    SetOffset(offset);
    return *this;
}

bool
QueryRequest::IgnoreGrowing() const {
    return GetExtraBool(extra_params_, "ignore_growing", false);
}

void
QueryRequest::SetIgnoreGrowing(bool ignore_growing) {
    SetExtraBool(extra_params_, "ignore_growing", ignore_growing);
}

QueryRequest&
QueryRequest::WithIgnoreGrowing(bool ignore_growing) {
    SetIgnoreGrowing(ignore_growing);
    return *this;
}

QueryRequest&
QueryRequest::AddExtraParam(const std::string& key, const std::string& value) {
    extra_params_[key] = value;
    return *this;
}

const std::unordered_map<std::string, std::string>&
QueryRequest::ExtraParams() const {
    return extra_params_;
}

}  // namespace milvus
