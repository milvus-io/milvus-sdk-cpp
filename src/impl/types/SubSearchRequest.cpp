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

#include <nlohmann/json.hpp>
#include <utility>

#include "../utils/Constants.h"

namespace milvus {

const std::string&
SubSearchRequest::Filter() const {
    return filter_expression_;
}

Status
SubSearchRequest::SetFilter(std::string filter) {
    filter_expression_ = std::move(filter);
    return Status::OK();
}

FieldDataPtr
SubSearchRequest::TargetVectors() const {
    return target_vectors_;
}

Status
SubSearchRequest::AddBinaryVector(std::string field_name, const std::string& vector) {
    return AddBinaryVector(std::move(field_name), BinaryVecFieldData::ToUnsignedChars(vector));
}

Status
SubSearchRequest::AddBinaryVector(std::string field_name, const BinaryVecFieldData::ElementT& vector) {
    auto status = verifyVectorType(DataType::BINARY_VECTOR);
    if (!status.IsOk()) {
        return status;
    }

    BinaryVecFieldDataPtr vectors;
    if (nullptr == target_vectors_) {
        vectors = std::make_shared<BinaryVecFieldData>(std::move(field_name));
        target_vectors_ = vectors;
    } else {
        if (field_name != target_vectors_->Name()) {
            std::string msg = "The vector field name must be the same! Previous name is " + vectors->Name();
            return {StatusCode::INVALID_AGUMENT, msg};
        }
        vectors = std::static_pointer_cast<BinaryVecFieldData>(target_vectors_);
    }

    auto code = vectors->Add(vector);
    if (code != StatusCode::OK) {
        return {code, "Failed to add binary vector"};
    }

    return Status::OK();
}

Status
SubSearchRequest::AddFloatVector(std::string field_name, const FloatVecFieldData::ElementT& vector) {
    auto status = verifyVectorType(DataType::FLOAT_VECTOR);
    if (!status.IsOk()) {
        return status;
    }

    FloatVecFieldDataPtr vectors;
    if (nullptr == target_vectors_) {
        vectors = std::make_shared<FloatVecFieldData>(std::move(field_name));
        target_vectors_ = vectors;
    } else {
        if (field_name != target_vectors_->Name()) {
            std::string msg = "The vector field name must be the same! Previous name is " + vectors->Name();
            return {StatusCode::INVALID_AGUMENT, msg};
        }
        vectors = std::static_pointer_cast<FloatVecFieldData>(target_vectors_);
    }

    auto code = vectors->Add(vector);
    if (code != StatusCode::OK) {
        return {code, "Failed to add float vector"};
    }

    return Status::OK();
}

Status
SubSearchRequest::AddSparseVector(std::string field_name, const SparseFloatVecFieldData::ElementT& vector) {
    auto status = verifyVectorType(DataType::SPARSE_FLOAT_VECTOR);
    if (!status.IsOk()) {
        return status;
    }

    SparseFloatVecFieldDataPtr vectors;
    if (nullptr == target_vectors_) {
        vectors = std::make_shared<SparseFloatVecFieldData>(std::move(field_name));
        target_vectors_ = vectors;
    } else {
        if (field_name != target_vectors_->Name()) {
            std::string msg = "The vector field name must be the same! Previous name is " + vectors->Name();
            return {StatusCode::INVALID_AGUMENT, msg};
        }
        vectors = std::static_pointer_cast<SparseFloatVecFieldData>(target_vectors_);
    }

    auto code = vectors->Add(vector);
    if (code != StatusCode::OK) {
        return {code, "Failed to add sparse vector"};
    }

    return Status::OK();
}

std::string
SubSearchRequest::AnnsField() const {
    if (target_vectors_ != nullptr) {
        return target_vectors_->Name();
    }
    return "";
}

int64_t
SubSearchRequest::Limit() const {
    return limit_;
}

Status
SubSearchRequest::SetLimit(int64_t limit) {
    limit_ = limit;
    return Status::OK();
}

::milvus::MetricType
SubSearchRequest::MetricType() const {
    return metric_type_;
}

Status
SubSearchRequest::SetMetricType(::milvus::MetricType metric_type) {
    // directly pass metric_type to server, no need to verify here
    metric_type_ = metric_type;
    return Status::OK();
}

Status
SubSearchRequest::AddExtraParam(const std::string& key, const std::string& value) {
    extra_params_[key] = value;
    static std::set<std::string> s_ambiguous = {KeyParams(),     KeyTopK(),         KeyAnnsField(),
                                                KeyMetricType(), KeyRoundDecimal(), KeyIgnoreGrowing()};
    if (s_ambiguous.find(key) != s_ambiguous.end()) {
        return Status{StatusCode::INVALID_AGUMENT,
                      "ambiguous parameter: not allow to set '" + key + "' in extra params"};
    }
    return Status::OK();
}

const std::unordered_map<std::string, std::string>&
SubSearchRequest::ExtraParams() const {
    return extra_params_;
}

Status
SubSearchRequest::Validate() const {
    // in milvus 2.4+, no need to check index parameters, let the server to check it
    if (target_vectors_ == nullptr || target_vectors_->Count() == 0) {
        return Status{StatusCode::INVALID_AGUMENT, "no target vector is assigned"};
    }
    return Status::OK();
}

float
SubSearchRequest::Radius() const {
    auto it = extra_params_.find(KeyRadius());
    if (it != extra_params_.end()) {
        return std::stof(it->second);
    }
    return 0;
}

Status
SubSearchRequest::SetRadius(float value) {
    extra_params_[KeyRadius()] = std::to_string(value);
    return Status::OK();
}

float
SubSearchRequest::RangeFilter() const {
    auto it = extra_params_.find(KeyRangeFilter());
    if (it != extra_params_.end()) {
        return std::stof(it->second);
    }
    return 0;
}

Status
SubSearchRequest::SetRangeFilter(float value) {
    extra_params_[KeyRangeFilter()] = std::to_string(value);
    return Status::OK();
}

Status
SubSearchRequest::SetRange(float range_filter, float radius) {
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
SubSearchRequest::verifyVectorType(DataType data_type) const {
    if (target_vectors_ != nullptr && target_vectors_->Type() != data_type) {
        return {StatusCode::INVALID_AGUMENT, "Target vector must be the same type!"};
    }
    return Status::OK();
}

}  // namespace milvus
