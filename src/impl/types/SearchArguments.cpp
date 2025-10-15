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
#include "../utils/DmlUtils.h"
#include "../utils/DqlUtils.h"
#include "../utils/TypeUtils.h"
#include "milvus/utils/FP16.h"

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

int64_t
SearchArguments::Offset() const {
    return offset_;
}

Status
SearchArguments::SetOffset(int64_t offset) {
    offset_ = offset;
    return Status::OK();
}

int
SearchArguments::RoundDecimal() const {
    return round_decimal_;
}

Status
SearchArguments::SetRoundDecimal(int round_decimal) {
    round_decimal_ = round_decimal;
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

std::string
SearchArguments::GroupByField() const {
    return group_by_field_;
}

Status
SearchArguments::SetGroupByField(const std::string& field_name) {
    group_by_field_ = field_name;
    return Status::OK();
}

Status
SearchArguments::Validate() const {
    // in milvus 2.4+, no need to check index parameters, let the server to check it
    if (target_vectors_ == nullptr || target_vectors_->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "no target vector is assigned"};
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
    auto it = extra_params_.find(NPROBE);
    if (it != extra_params_.end()) {
        return std::stoll(it->second);
    }
    return 0;
}

Status
SearchArguments::SetNprobe(int64_t nprobe) {
    extra_params_[NPROBE] = std::to_string(nprobe);
    return Status::OK();
}

uint64_t
SearchArguments::TravelTimestamp() const {
    return 0;
}

Status
SearchArguments::SetTravelTimestamp(uint64_t timestamp) {
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
