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

#include "milvus/request/dql/HybridSearchRequest.h"

#include <memory>

#include "../../utils/ExtraParamUtils.h"

namespace milvus {

HybridSearchRequest&
HybridSearchRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

HybridSearchRequest&
HybridSearchRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

HybridSearchRequest&
HybridSearchRequest::WithPartitionNames(std::set<std::string>&& partition_names) {
    SetPartitionNames(std::move(partition_names));
    return *this;
}

HybridSearchRequest&
HybridSearchRequest::AddPartitionName(const std::string& partition_name) {
    DQLRequestBase::AddPartitionName(partition_name);
    return *this;
}

HybridSearchRequest&
HybridSearchRequest::WithOutputFields(std::set<std::string>&& output_field_names) {
    SetOutputFields(std::move(output_field_names));
    return *this;
}

HybridSearchRequest&
HybridSearchRequest::AddOutputField(const std::string& output_field) {
    DQLRequestBase::AddOutputField(output_field);
    return *this;
}

HybridSearchRequest&
HybridSearchRequest::WithConsistencyLevel(ConsistencyLevel consistency_level) {
    SetConsistencyLevel(consistency_level);
    return *this;
}

const std::vector<SubSearchRequestPtr>&
HybridSearchRequest::SubRequests() const {
    return sub_requests_;
}

void
HybridSearchRequest::SetSubRequest(std::vector<SubSearchRequestPtr>&& requests) {
    sub_requests_ = std::move(requests);
}

HybridSearchRequest&
HybridSearchRequest::WithSubRequest(std::vector<SubSearchRequestPtr>&& requests) {
    sub_requests_ = std::move(requests);
    return *this;
}

HybridSearchRequest&
HybridSearchRequest::AddSubRequest(SubSearchRequestPtr&& request) {
    sub_requests_.emplace_back(request);
    return *this;
}

FunctionPtr
HybridSearchRequest::Rerank() const {
    return function_;
}

Status
HybridSearchRequest::SetRerank(const FunctionPtr& rerank) {
    function_ = rerank;
    return Status::OK();
}

HybridSearchRequest&
HybridSearchRequest::WithRerank(const FunctionPtr& rerank) {
    function_ = rerank;
    return *this;
}

int64_t
HybridSearchRequest::Limit() const {
    return limit_;
}

Status
HybridSearchRequest::SetLimit(int64_t limit) {
    limit_ = limit;
    return Status::OK();
}

HybridSearchRequest&
HybridSearchRequest::WithLimit(int64_t limit) {
    limit_ = limit;
    return *this;
}

int64_t
HybridSearchRequest::Offset() const {
    return GetExtraInt64(extra_params_, "offset", 0);
}

void
HybridSearchRequest::SetOffset(int64_t offset) {
    SetExtraInt64(extra_params_, "offset", offset);
}

HybridSearchRequest&
HybridSearchRequest::WithOffset(int64_t offset) {
    SetOffset(offset);
    return *this;
}

int64_t
HybridSearchRequest::GetRoundDecimal() const {
    return GetExtraInt64(extra_params_, "round_decimal", -1);
}

void
HybridSearchRequest::SetRoundDecimal(int64_t round_decimal) {
    SetExtraInt64(extra_params_, "round_decimal", round_decimal);
}

HybridSearchRequest&
HybridSearchRequest::WithRoundDecimal(int64_t round_decimal) {
    SetRoundDecimal(round_decimal);
    return *this;
}

bool
HybridSearchRequest::IgnoreGrowing() const {
    return GetExtraBool(extra_params_, "ignore_growing", false);
}

void
HybridSearchRequest::SetIgnoreGrowing(bool ignore_growing) {
    SetExtraBool(extra_params_, "ignore_growing", ignore_growing);
}

HybridSearchRequest&
HybridSearchRequest::WithIgnoreGrowing(bool ignore_growing) {
    SetIgnoreGrowing(ignore_growing);
    return *this;
}

HybridSearchRequest&
HybridSearchRequest::AddExtraParam(const std::string& key, const std::string& value) {
    extra_params_[key] = value;
    return *this;
}

const std::unordered_map<std::string, std::string>&
HybridSearchRequest::ExtraParams() const {
    return extra_params_;
}

std::string
HybridSearchRequest::GetGroupByField() const {
    return GetExtraStr(extra_params_, "group_by_field", "");
}

void
HybridSearchRequest::SetGroupByField(const std::string& field_name) {
    SetExtraStr(extra_params_, "group_by_field", field_name);
}

HybridSearchRequest&
HybridSearchRequest::WithGroupByField(const std::string& field_name) {
    SetGroupByField(field_name);
    return *this;
}

int64_t
HybridSearchRequest::GroupSize() const {
    return GetExtraInt64(extra_params_, "group_size", 1);
}

void
HybridSearchRequest::SetGroupSize(int64_t group_size) {
    SetExtraInt64(extra_params_, "group_size", group_size);
}

HybridSearchRequest&
HybridSearchRequest::WithGroupSize(int64_t group_size) {
    SetGroupSize(group_size);
    return *this;
}

bool
HybridSearchRequest::StrictGroupSize() const {
    return GetExtraBool(extra_params_, "strict_group_size", false);
}

void
HybridSearchRequest::SetStrictGroupSize(bool strict_group_size) {
    SetExtraBool(extra_params_, "strict_group_size", strict_group_size);
}

HybridSearchRequest&
HybridSearchRequest::WithStrictGroupSize(bool strict_group_size) {
    SetStrictGroupSize(strict_group_size);
    return *this;
}

}  // namespace milvus
