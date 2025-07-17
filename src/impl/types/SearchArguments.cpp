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

#include "milvus/types/SearchArguments.h"

#include <nlohmann/json.hpp>
#include <utility>

#include "../utils/Constants.h"

namespace milvus {

const std::string&
SearchArguments::DatabaseName() const {
    return db_name_;
}

Status
SearchArguments::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
    return Status::OK();
}

const std::string&
SearchArguments::CollectionName() const {
    return collection_name_;
}

Status
SearchArguments::SetCollectionName(std::string collection_name) {
    if (collection_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Collection name cannot be empty!"};
    }
    collection_name_ = std::move(collection_name);
    return Status::OK();
}

const std::set<std::string>&
SearchArguments::PartitionNames() const {
    return partition_names_;
}

Status
SearchArguments::AddPartitionName(std::string partition_name) {
    if (partition_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Partition name cannot be empty!"};
    }
    partition_names_.emplace(std::move(partition_name));
    return Status::OK();
}

const std::set<std::string>&
SearchArguments::OutputFields() const {
    return output_field_names_;
}

Status
SearchArguments::AddOutputField(std::string field_name) {
    if (field_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
    }

    output_field_names_.emplace(std::move(field_name));
    return Status::OK();
}

const std::string&
SearchArguments::Filter() const {
    return filter_expression_;
}

Status
SearchArguments::SetFilter(std::string filter) {
    filter_expression_ = std::move(filter);
    return Status::OK();
}

FieldDataPtr
SearchArguments::TargetVectors() const {
    return target_vectors_;
}

Status
SearchArguments::AddBinaryVector(std::string field_name, const std::vector<uint8_t>& vector) {
    return AddBinaryVector(std::move(field_name), milvus::BinaryVecFieldData::CreateBinaryString(vector));
}

Status
SearchArguments::AddBinaryVector(std::string field_name, const BinaryVecFieldData::ElementT& vector) {
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
SearchArguments::AddFloatVector(std::string field_name, const FloatVecFieldData::ElementT& vector) {
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
SearchArguments::AddSparseVector(std::string field_name, const SparseFloatVecFieldData::ElementT& vector) {
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
SearchArguments::AnnsField() const {
    if (target_vectors_ != nullptr) {
        return target_vectors_->Name();
    }
    return "";
}

int64_t
SearchArguments::Limit() const {
    return limit_;
}

Status
SearchArguments::SetLimit(int64_t limit) {
    limit_ = limit;
    return Status::OK();
}

int64_t
SearchArguments::Offset() const {
    auto it = extra_params_.find(KeyOffset());
    if (it != extra_params_.end()) {
        return std::stoll(it->second);
    }
    return 0;
}

Status
SearchArguments::SetOffset(int64_t offset) {
    extra_params_[KeyOffset()] = std::to_string(offset);
    return Status::OK();
}

Status
SearchArguments::SetRoundDecimal(int round_decimal) {
    round_decimal_ = round_decimal;
    return Status::OK();
}

int
SearchArguments::RoundDecimal() const {
    return round_decimal_;
}

Status
SearchArguments::SetMetricType(::milvus::MetricType metric_type) {
    // directly pass metric_type to server, no need to verify here
    metric_type_ = metric_type;
    return Status::OK();
}

::milvus::MetricType
SearchArguments::MetricType() const {
    return metric_type_;
}

Status
SearchArguments::AddExtraParam(const std::string& key, const std::string& value) {
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
SearchArguments::ExtraParams() const {
    return extra_params_;
}

Status
SearchArguments::Validate() const {
    // in milvus 2.4+, no need to check index parameters, let the server to check it
    if (target_vectors_ == nullptr || target_vectors_->Count() == 0) {
        return Status{StatusCode::INVALID_AGUMENT, "no target vector is assigned"};
    }
    return Status::OK();
}

float
SearchArguments::Radius() const {
    auto it = extra_params_.find(KeyRadius());
    if (it != extra_params_.end()) {
        return std::stof(it->second);
    }
    return 0;
}

Status
SearchArguments::SetRadius(float value) {
    extra_params_[KeyRadius()] = std::to_string(value);
    return Status::OK();
}

float
SearchArguments::RangeFilter() const {
    auto it = extra_params_.find(KeyRangeFilter());
    if (it != extra_params_.end()) {
        return std::stof(it->second);
    }
    return 0;
}

Status
SearchArguments::SetRangeFilter(float value) {
    extra_params_[KeyRangeFilter()] = std::to_string(value);
    return Status::OK();
}

Status
SearchArguments::SetRange(float range_filter, float radius) {
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

ConsistencyLevel
SearchArguments::GetConsistencyLevel() const {
    return consistency_level_;
}

Status
SearchArguments::SetConsistencyLevel(const ConsistencyLevel& level) {
    consistency_level_ = level;
    return Status::OK();
}

bool
SearchArguments::IgnoreGrowing() const {
    return ignore_growing_;
}

Status
SearchArguments::SetIgnoreGrowing(bool ignore_growing) {
    ignore_growing_ = ignore_growing;
    return Status::OK();
}

Status
SearchArguments::verifyVectorType(DataType data_type) const {
    if (target_vectors_ != nullptr && target_vectors_->Type() != data_type) {
        return {StatusCode::INVALID_AGUMENT, "Target vector must be the same type!"};
    }
    return Status::OK();
}

///////////////////////////////////////////////////////////////////////////////////////
// deprecated methods
const std::string&
SearchArguments::Expression() const {
    return Filter();
}

Status
SearchArguments::SetExpression(std::string expression) {
    return SetFilter(expression);
}

Status
SearchArguments::AddTargetVector(std::string field_name, const std::string& vector) {
    return AddBinaryVector(std::move(field_name), vector);
}

Status
SearchArguments::AddTargetVector(std::string field_name, const std::vector<uint8_t>& vector) {
    return AddBinaryVector(std::move(field_name), vector);
}

Status
SearchArguments::AddTargetVector(std::string field_name, std::string&& vector) {
    return AddBinaryVector(std::move(field_name), vector);
}

Status
SearchArguments::AddTargetVector(std::string field_name, const FloatVecFieldData::ElementT& vector) {
    return AddFloatVector(std::move(field_name), vector);
}

Status
SearchArguments::AddTargetVector(std::string field_name, FloatVecFieldData::ElementT&& vector) {
    return AddFloatVector(std::move(field_name), vector);
}

Status
SearchArguments::SetTopK(int64_t topk) {
    return SetLimit(topk);
}

int64_t
SearchArguments::TopK() const {
    return Limit();
}

int64_t
SearchArguments::Nprobe() const {
    auto it = extra_params_.find(KeyNprobe());
    if (it != extra_params_.end()) {
        return std::stoll(it->second);
    }
    return 0;
}

Status
SearchArguments::SetNprobe(int64_t nprobe) {
    extra_params_[KeyNprobe()] = std::to_string(nprobe);
    return Status::OK();
}

uint64_t
SearchArguments::TravelTimestamp() const {
    return travel_timestamp_;
}

Status
SearchArguments::SetTravelTimestamp(uint64_t timestamp) {
    travel_timestamp_ = timestamp;
    return Status::OK();
}

uint64_t
SearchArguments::GuaranteeTimestamp() const {
    return 0;
}

Status
SearchArguments::SetGuaranteeTimestamp(uint64_t timestamp) {
    return Status::OK();
}
///////////////////////////////////////////////////////////////////////////////////////

}  // namespace milvus
