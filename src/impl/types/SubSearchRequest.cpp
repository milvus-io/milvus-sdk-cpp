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

#include "milvus/types/SubSearchRequest.h"

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
    return target_vectors_;
}

Status
SearchRequestBase::AddBinaryVector(std::string field_name, const std::string& vector) {
    return AddBinaryVector(std::move(field_name), BinaryVecFieldData::ToUnsignedChars(vector));
}

Status
SearchRequestBase::AddBinaryVector(std::string field_name, const BinaryVecFieldData::ElementT& vector) {
    return addVector<BinaryVecFieldData, BinaryVecFieldData::ElementT>(field_name, DataType::BINARY_VECTOR, vector);
}

Status
SearchRequestBase::AddFloatVector(std::string field_name, const FloatVecFieldData::ElementT& vector) {
    return addVector<FloatVecFieldData, FloatVecFieldData::ElementT>(field_name, DataType::FLOAT_VECTOR, vector);
}

Status
SearchRequestBase::AddSparseVector(std::string field_name, const SparseFloatVecFieldData::ElementT& vector) {
    return addVector<SparseFloatVecFieldData, SparseFloatVecFieldData::ElementT>(field_name,
                                                                                 DataType::SPARSE_FLOAT_VECTOR, vector);
}

Status
SearchRequestBase::AddSparseVector(std::string field_name, const nlohmann::json& vector) {
    std::map<uint32_t, float> pairs;
    auto status = ParseSparseFloatVector(vector, field_name, pairs);
    if (!status.IsOk()) {
        return status;
    }
    return AddSparseVector(field_name, pairs);
}

Status
SearchRequestBase::AddFloat16Vector(std::string field_name, const Float16VecFieldData::ElementT& vector) {
    return addVector<Float16VecFieldData, Float16VecFieldData::ElementT>(field_name, DataType::FLOAT16_VECTOR, vector);
}

Status
SearchRequestBase::AddFloat16Vector(std::string field_name, const std::vector<float>& vector) {
    std::vector<uint16_t> binary;
    binary.reserve(vector.size());
    for (auto val : vector) {
        binary.push_back(F32toF16(val));
    }
    return AddFloat16Vector(field_name, binary);
}

Status
SearchRequestBase::AddBFloat16Vector(std::string field_name, const BFloat16VecFieldData::ElementT& vector) {
    return addVector<BFloat16VecFieldData, BFloat16VecFieldData::ElementT>(field_name, DataType::BFLOAT16_VECTOR,
                                                                           vector);
}

Status
SearchRequestBase::AddBFloat16Vector(std::string field_name, const std::vector<float>& vector) {
    std::vector<uint16_t> binary;
    binary.reserve(vector.size());
    for (auto val : vector) {
        binary.push_back(F32toBF16(val));
    }
    return AddBFloat16Vector(field_name, binary);
}

Status
SearchRequestBase::AddEmbeddedText(std::string field_name, const std::string& text) {
    return addVector<VarCharFieldData, VarCharFieldData::ElementT>(field_name, DataType::VARCHAR, text);
}

Status
SearchRequestBase::AddInt8Vector(std::string field_name, const Int8VecFieldData::ElementT& vector) {
    return addVector<Int8VecFieldData, Int8VecFieldData::ElementT>(field_name, DataType::INT8_VECTOR, vector);
}

std::string
SearchRequestBase::AnnsField() const {
    if (target_vectors_ != nullptr) {
        return target_vectors_->Name();
    }
    return "";
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
    if (target_vectors_ == nullptr || target_vectors_->Count() == 0) {
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

Status
SearchRequestBase::verifyVectorType(DataType data_type) const {
    if (target_vectors_ != nullptr && target_vectors_->Type() != data_type) {
        return {StatusCode::INVALID_AGUMENT, "Target vector must be the same type!"};
    }
    return Status::OK();
}

template <typename T, typename V>
Status
SearchRequestBase::addVector(std::string field_name, DataType data_type, const V& vector) {
    auto status = verifyVectorType(data_type);
    if (!status.IsOk()) {
        return status;
    }

    StatusCode code = StatusCode::OK;
    if (nullptr == target_vectors_) {
        std::shared_ptr<T> vectors = std::make_shared<T>(std::move(field_name));
        target_vectors_ = vectors;
        code = vectors->Add(vector);
    } else {
        if (field_name != target_vectors_->Name()) {
            std::string msg = "The vector field name must be the same! Previous name is " + target_vectors_->Name();
            return {StatusCode::INVALID_AGUMENT, msg};
        }
        std::shared_ptr<T> vectors = std::static_pointer_cast<T>(target_vectors_);
        code = vectors->Add(vector);
    }

    if (code != StatusCode::OK) {
        return {code, "Failed to add " + std::to_string(data_type)};
    }

    return Status::OK();
}

}  // namespace milvus
