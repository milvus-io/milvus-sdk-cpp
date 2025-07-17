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

#include "milvus/types/IndexDesc.h"

#include <nlohmann/json.hpp>

#include "../utils/TypeUtils.h"

namespace milvus {

IndexDesc::IndexDesc() = default;

IndexDesc::IndexDesc(std::string field_name, std::string index_name, milvus::IndexType index_type,
                     milvus::MetricType metric_type)
    : field_name_(std::move(field_name)),
      index_name_(std::move(index_name)),
      index_type_(index_type),
      metric_type_(metric_type) {
}

const std::string&
IndexDesc::FieldName() const {
    return field_name_;
}

Status
IndexDesc::SetFieldName(std::string field_name) {
    field_name_ = std::move(field_name);
    return Status::OK();
}

const std::string&
IndexDesc::IndexName() const {
    return index_name_;
}

Status
IndexDesc::SetIndexName(std::string index_name) {
    index_name_ = std::move(index_name);
    return Status::OK();
}

int64_t
IndexDesc::IndexId() const {
    return index_id_;
}

Status
IndexDesc::SetIndexId(int64_t index_id) {
    index_id_ = index_id;
    return Status::OK();
}

milvus::MetricType
IndexDesc::MetricType() const {
    return metric_type_;
}

Status
IndexDesc::SetMetricType(milvus::MetricType metric_type) {
    metric_type_ = metric_type;
    return Status::OK();
}

milvus::IndexType
IndexDesc::IndexType() const {
    return index_type_;
}

Status
IndexDesc::SetIndexType(milvus::IndexType index_type) {
    index_type_ = index_type;
    return Status::OK();
}

Status
IndexDesc::AddExtraParam(const std::string& key, const std::string& value) {
    extra_params_[key] = value;
    return Status::OK();
}

const std::unordered_map<std::string, std::string>&
IndexDesc::ExtraParams() const {
    return extra_params_;
}

Status
IndexDesc::ExtraParamsFromJson(std::string json) {
    try {
        extra_params_ = ::nlohmann::json::parse(std::move(json));
    } catch (const ::nlohmann::json::exception& e) {
        return {StatusCode::JSON_PARSE_ERROR, e.what()};
    }
    return Status::OK();
}

Status
IndexDesc::SetStateCode(const milvus::IndexStateCode& code) {
    state_code_ = code;
    return Status::OK();
}

milvus::IndexStateCode
IndexDesc::StateCode() const {
    return state_code_;
}

Status
IndexDesc::SetFailReason(const std::string& reason) {
    failed_reason_ = reason;
    return Status::OK();
}

std::string
IndexDesc::FailReason() const {
    return failed_reason_;
}

Status
IndexDesc::SetIndexedRows(int64_t rows) {
    indexed_rows_ = rows;
    return Status::OK();
}

int64_t
IndexDesc::IndexedRows() const {
    return indexed_rows_;
}

Status
IndexDesc::SetTotalRows(int64_t rows) {
    total_rows_ = rows;
    return Status::OK();
}

int64_t
IndexDesc::TotalRows() const {
    return total_rows_;
}

Status
IndexDesc::SetPendingRows(int64_t rows) {
    pending_rows_ = rows;
    return Status::OK();
}

int64_t
IndexDesc::PendingRows() const {
    return pending_rows_;
}

}  // namespace milvus
