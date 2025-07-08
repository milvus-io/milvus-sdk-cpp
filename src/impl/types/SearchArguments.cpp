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
SearchArguments::Expression() const {
    return filter_expression_;
}

Status
SearchArguments::SetExpression(std::string expression) {
    filter_expression_ = std::move(expression);
    return Status::OK();
}

FieldDataPtr
SearchArguments::TargetVectors() const {
    if (binary_vectors_ != nullptr) {
        return binary_vectors_;
    } else if (float_vectors_ != nullptr) {
        return float_vectors_;
    }

    return nullptr;
}

Status
SearchArguments::AddTargetVector(std::string field_name, const std::string& vector) {
    return AddTargetVector(std::move(field_name), std::string{vector});
}

Status
SearchArguments::AddTargetVector(std::string field_name, const std::vector<uint8_t>& vector) {
    return AddTargetVector(std::move(field_name), milvus::BinaryVecFieldData::CreateBinaryString(vector));
}

Status
SearchArguments::AddTargetVector(std::string field_name, std::string&& vector) {
    if (float_vectors_ != nullptr) {
        return {StatusCode::INVALID_AGUMENT, "Target vector must be float type!"};
    }

    if (nullptr == binary_vectors_) {
        binary_vectors_ = std::make_shared<BinaryVecFieldData>(std::move(field_name));
    }

    auto code = binary_vectors_->Add(std::move(vector));
    if (code != StatusCode::OK) {
        return {code, "Failed to add vector"};
    }

    return Status::OK();
}

Status
SearchArguments::AddTargetVector(std::string field_name, const FloatVecFieldData::ElementT& vector) {
    if (binary_vectors_ != nullptr) {
        return {StatusCode::INVALID_AGUMENT, "Target vector must be binary type!"};
    }

    if (nullptr == float_vectors_) {
        float_vectors_ = std::make_shared<FloatVecFieldData>(std::move(field_name));
    }

    auto code = float_vectors_->Add(vector);
    if (code != StatusCode::OK) {
        return {code, "Failed to add vector"};
    }

    return Status::OK();
}

Status
SearchArguments::AddTargetVector(std::string field_name, FloatVecFieldData::ElementT&& vector) {
    if (binary_vectors_ != nullptr) {
        return {StatusCode::INVALID_AGUMENT, "Target vector must be binary type!"};
    }

    if (nullptr == float_vectors_) {
        float_vectors_ = std::make_shared<FloatVecFieldData>(std::move(field_name));
    }

    auto code = float_vectors_->Add(std::move(vector));
    if (code != StatusCode::OK) {
        return {code, "Failed to add vector"};
    }

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
    return guarantee_timestamp_;
}

Status
SearchArguments::SetGuaranteeTimestamp(uint64_t timestamp) {
    guarantee_timestamp_ = timestamp;
    return Status::OK();
}

Status
SearchArguments::SetTopK(int64_t topk) {
    topk_ = topk;
    return Status::OK();
}

int64_t
SearchArguments::TopK() const {
    return topk_;
}

int64_t
SearchArguments::Nprobe() const {
    if (extra_params_.find("nprobe") != extra_params_.end()) {
        return extra_params_.at("nprobe");
    }
    return 1;
}

Status
SearchArguments::SetNprobe(int64_t nprobe) {
    extra_params_["nprobe"] = nprobe;
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
SearchArguments::AddExtraParam(std::string key, int64_t value) {
    extra_params_[std::move(key)] = value;
    return Status::OK();
}

std::string
SearchArguments::ExtraParams() const {
    return ::nlohmann::json(extra_params_).dump();
}

Status
SearchArguments::Validate(std::string& anns_field) const {
    // in milvus 2.4+, no need to check index parameters, let the server to check it

    auto float_vector_count = (float_vectors_ == nullptr) ? 0 : float_vectors_->Count();
    auto binary_vector_count = (binary_vectors_ == nullptr) ? 0 : binary_vectors_->Count();
    // no target vector is assigned, not able search
    if (float_vector_count == 0 && binary_vector_count == 0) {
        return Status{StatusCode::INVALID_AGUMENT, "no target vector is assigned"};
    }
    // all the target vectors must be the same type
    if (float_vector_count > 0 && binary_vector_count > 0) {
        return Status{StatusCode::INVALID_AGUMENT, "target vectors are not same type"};
    }

    // return anns_field name
    if (float_vector_count > 0) {
        anns_field = float_vectors_->Name();
    } else if (binary_vector_count > 0) {
        anns_field = binary_vectors_->Name();
    }
    return Status::OK();
}

float
SearchArguments::Radius() const {
    return radius_;
}

float
SearchArguments::RangeFilter() const {
    return range_filter_;
}

Status
SearchArguments::SetRange(float range_filter, float radius) {
    // directly pass the radius/range_filter to let server validate, no need to verify here
    radius_ = radius;
    range_filter_ = range_filter;
    range_search_ = true;

    return Status::OK();
}

bool
SearchArguments::RangeSearch() const {
    return range_search_;
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

}  // namespace milvus
