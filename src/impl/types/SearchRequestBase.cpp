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

#include "milvus/types/SearchRequestBase.h"

#include <utility>

#include "../utils/Constants.h"
#include "../utils/DmlUtils.h"
#include "../utils/DqlUtils.h"
#include "../utils/ExtraParamUtils.h"
#include "../utils/TypeUtils.h"
#include "milvus/utils/FP16.h"

namespace milvus {
const std::string&
SearchRequestBase::Filter() const {
    return filter_expression_;
}

Status
SearchRequestBase::SetFilter(std::string filter) {
    filter_expression_ = std::move(filter);
    return Status::OK();
}

Status
SearchRequestBase::AddFilterTemplate(std::string key, const nlohmann::json& filter_template) {
    if (filter_template.is_array()) {
        for (const auto& ele : filter_template) {
            if (!IsValidTemplate(ele)) {
                return {milvus::StatusCode::INVALID_AGUMENT, "Filter template element must be boolean/number/string"};
            }
        }
    } else {
        if (!IsValidTemplate(filter_template)) {
            return {milvus::StatusCode::INVALID_AGUMENT, "Filter template must be boolean/number/string/array"};
        }
    }

    filter_templates_.insert(std::make_pair(key, filter_template));
    return Status::OK();
}

const std::unordered_map<std::string, nlohmann::json>&
SearchRequestBase::FilterTemplates() const {
    return filter_templates_;
}

Status
SearchRequestBase::SetFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates) {
    filter_templates_ = std::move(filter_templates);
    return Status::OK();
}

FieldDataPtr
SearchRequestBase::TargetVectors() const {
    return target_vectors_.TargetVectors();
}

Status
SearchRequestBase::AddBinaryVector(const std::string& vector) {
    return target_vectors_.AddBinaryVector(vector);
}

Status
SearchRequestBase::AddBinaryVector(const BinaryVecFieldData::ElementT& vector) {
    return target_vectors_.AddBinaryVector(vector);
}

Status
SearchRequestBase::AddFloatVector(const FloatVecFieldData::ElementT& vector) {
    return target_vectors_.AddFloatVector(vector);
}

Status
SearchRequestBase::AddSparseVector(const SparseFloatVecFieldData::ElementT& vector) {
    return target_vectors_.AddSparseVector(vector);
}

Status
SearchRequestBase::AddSparseVector(const nlohmann::json& vector) {
    return target_vectors_.AddSparseVector(vector);
}

Status
SearchRequestBase::AddFloat16Vector(const Float16VecFieldData::ElementT& vector) {
    return target_vectors_.AddFloat16Vector(vector);
}

Status
SearchRequestBase::AddFloat16Vector(const std::vector<float>& vector) {
    return target_vectors_.AddFloat16Vector(vector);
}

Status
SearchRequestBase::AddBFloat16Vector(const BFloat16VecFieldData::ElementT& vector) {
    return target_vectors_.AddBFloat16Vector(vector);
}

Status
SearchRequestBase::AddBFloat16Vector(const std::vector<float>& vector) {
    return target_vectors_.AddBFloat16Vector(vector);
}

Status
SearchRequestBase::AddEmbeddedText(const std::string& text) {
    return target_vectors_.AddEmbeddedText(text);
}

std::string
SearchRequestBase::AnnsField() const {
    return ann_field_;
}

Status
SearchRequestBase::SetAnnsField(const std::string& ann_field) {
    ann_field_ = ann_field;
    return Status::OK();
}

int64_t
SearchRequestBase::Limit() const {
    return limit_;
}

Status
SearchRequestBase::SetLimit(int64_t limit) {
    limit_ = limit;
    return Status::OK();
}

::milvus::MetricType
SearchRequestBase::MetricType() const {
    return metric_type_;
}

Status
SearchRequestBase::SetMetricType(::milvus::MetricType metric_type) {
    // directly pass metric_type to server, no need to verify here
    metric_type_ = metric_type;
    return Status::OK();
}

Status
SearchRequestBase::AddExtraParam(const std::string& key, const std::string& value) {
    auto status = IsAmbiguousParam(key);
    if (status.IsOk()) {
        extra_params_[key] = value;
    }
    return status;
}

const std::unordered_map<std::string, std::string>&
SearchRequestBase::ExtraParams() const {
    return extra_params_;
}

Status
SearchRequestBase::Validate() const {
    // in milvus 2.4+, no need to check index parameters, let the server to check it
    if (target_vectors_.Count() == 0) {
        return Status{StatusCode::INVALID_AGUMENT, "no target vector is assigned"};
    }
    return Status::OK();
}

double
SearchRequestBase::Radius() const {
    auto it = extra_params_.find(RADIUS);
    if (it != extra_params_.end()) {
        return std::stod(it->second);
    }
    return 0;
}

Status
SearchRequestBase::SetRadius(double value) {
    extra_params_[RADIUS] = DoubleToString(value);
    return Status::OK();
}

double
SearchRequestBase::RangeFilter() const {
    auto it = extra_params_.find(RANGE_FILTER);
    if (it != extra_params_.end()) {
        return std::stod(it->second);
    }
    return 0;
}

Status
SearchRequestBase::SetRangeFilter(double value) {
    extra_params_[RANGE_FILTER] = DoubleToString(value);
    return Status::OK();
}

Status
SearchRequestBase::SetRange(double range_filter, double radius) {
    // directly pass the radius/range_filter to let server validate, no need to verify here
    auto status = SetRadius(radius);
    if (!status.IsOk()) {
        return status;
    }
    status = SetRangeFilter(range_filter);
    if (!status.IsOk()) {
        return status;
    }

    return Status::OK();
}

///////////////////////////////////////////////////////////////////////////////////////
// deprecated methods
Status
SearchRequestBase::AddBinaryVector(std::string field_name, const std::string& vector) {
    SetAnnsField(field_name);
    return target_vectors_.AddBinaryVector(vector);
}

Status
SearchRequestBase::AddBinaryVector(std::string field_name, const BinaryVecFieldData::ElementT& vector) {
    SetAnnsField(field_name);
    return target_vectors_.AddBinaryVector(vector);
}

Status
SearchRequestBase::AddFloatVector(std::string field_name, const FloatVecFieldData::ElementT& vector) {
    SetAnnsField(field_name);
    return target_vectors_.AddFloatVector(vector);
}

Status
SearchRequestBase::AddSparseVector(std::string field_name, const SparseFloatVecFieldData::ElementT& vector) {
    SetAnnsField(field_name);
    return target_vectors_.AddSparseVector(vector);
}

Status
SearchRequestBase::AddSparseVector(std::string field_name, const nlohmann::json& vector) {
    SetAnnsField(field_name);
    return target_vectors_.AddSparseVector(vector);
}

Status
SearchRequestBase::AddFloat16Vector(std::string field_name, const Float16VecFieldData::ElementT& vector) {
    SetAnnsField(field_name);
    return target_vectors_.AddFloat16Vector(vector);
}

Status
SearchRequestBase::AddFloat16Vector(std::string field_name, const std::vector<float>& vector) {
    SetAnnsField(field_name);
    return target_vectors_.AddFloat16Vector(vector);
}

Status
SearchRequestBase::AddBFloat16Vector(std::string field_name, const BFloat16VecFieldData::ElementT& vector) {
    SetAnnsField(field_name);
    return target_vectors_.AddBFloat16Vector(vector);
}

Status
SearchRequestBase::AddBFloat16Vector(std::string field_name, const std::vector<float>& vector) {
    SetAnnsField(field_name);
    return target_vectors_.AddBFloat16Vector(vector);
}

Status
SearchRequestBase::AddEmbeddedText(std::string field_name, const std::string& text) {
    SetAnnsField(field_name);
    return target_vectors_.AddEmbeddedText(text);
}

}  // namespace milvus
