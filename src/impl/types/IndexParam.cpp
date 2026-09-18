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

#include "milvus/types/IndexParam.h"

#include <milvus/thirdparty/nlohmann/json.hpp>

namespace milvus {

IndexParam::IndexParam() = default;

IndexParam::IndexParam(std::string field_name, std::string index_name, milvus::IndexType index_type,
                       milvus::MetricType metric_type)
    : field_name_(std::move(field_name)),
      index_name_(std::move(index_name)),
      metric_type_(metric_type),
      index_type_(index_type) {
}

const std::string&
IndexParam::FieldName() const {
    return field_name_;
}

Status
IndexParam::SetFieldName(std::string field_name) {
    field_name_ = std::move(field_name);
    return Status::OK();
}

const std::string&
IndexParam::IndexName() const {
    return index_name_;
}

Status
IndexParam::SetIndexName(std::string index_name) {
    index_name_ = std::move(index_name);
    return Status::OK();
}

milvus::MetricType
IndexParam::MetricType() const {
    return metric_type_;
}

Status
IndexParam::SetMetricType(milvus::MetricType metric_type) {
    metric_type_ = metric_type;
    return Status::OK();
}

milvus::IndexType
IndexParam::IndexType() const {
    return index_type_;
}

Status
IndexParam::SetIndexType(milvus::IndexType index_type) {
    index_type_ = index_type;
    return Status::OK();
}

Status
IndexParam::AddExtraParam(const std::string& key, const std::string& value) {
    extra_params_[key] = value;
    return Status::OK();
}

const std::unordered_map<std::string, std::string>&
IndexParam::ExtraParams() const {
    return extra_params_;
}

Status
IndexParam::ExtraParamsFromJson(std::string json) {
    try {
        std::unordered_map<std::string, std::string> temp = ::nlohmann::json::parse(std::move(json));
        for (const auto& pair : temp) {
            extra_params_.insert(pair);
        }
    } catch (const ::nlohmann::json::exception& e) {
        return {StatusCode::JSON_PARSE_ERROR, e.what()};
    }
    return Status::OK();
}

}  // namespace milvus
