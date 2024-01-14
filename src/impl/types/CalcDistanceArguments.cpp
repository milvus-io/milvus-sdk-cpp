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

#include "milvus/types/CalcDistanceArguments.h"

#include <set>
#include <unordered_map>

#include "../TypeUtils.h"

namespace milvus {

namespace {
template <typename VectorPtr>
Status
validate_vectors_with_oper(const VectorPtr& vectors, std::function<void()> func) {
    if (nullptr == vectors || vectors->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
    }
    func();
    return Status::OK();
}

template <typename Name>
Status
validate_vectors_with_oper(const Int64FieldDataPtr& ids, Name name, std::function<void()> func) {
    if (nullptr == ids || ids->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
    }

    if (ids->Name().empty()) {
        return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
    }

    if (name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Collection name cannot be empty!"};
    }
    func();
    return Status::OK();
}
}  // namespace

CalcDistanceArguments::CalcDistanceArguments() = default;

Status
CalcDistanceArguments::SetLeftVectors(FloatVecFieldDataPtr vectors) {
    return validate_vectors_with_oper(vectors, [this, &vectors]() { this->vectors_left_ = std::move(vectors); });
}

Status
CalcDistanceArguments::SetLeftVectors(BinaryVecFieldDataPtr vectors) {
    return validate_vectors_with_oper(vectors, [this, &vectors]() { this->vectors_left_ = std::move(vectors); });
}

Status
CalcDistanceArguments::SetLeftVectors(Int64FieldDataPtr ids, std::string collection_name,
                                      const std::vector<std::string>& partition_names) {
    return validate_vectors_with_oper(ids, collection_name, [this, &ids, &collection_name, &partition_names]() {
        this->vectors_left_ = std::move(ids);
        this->collection_left_ = std::move(collection_name);
        this->partitions_left_ = partition_names;
    });
}

Status
CalcDistanceArguments::SetLeftVectors(Int64FieldDataPtr ids, std::string collection_name,
                                      std::vector<std::string>&& partition_names) {
    return validate_vectors_with_oper(ids, collection_name, [this, &ids, &collection_name, &partition_names]() {
        this->vectors_left_ = std::move(ids);
        this->collection_left_ = std::move(collection_name);
        this->partitions_left_ = std::move(partition_names);
    });
}

FieldDataPtr
CalcDistanceArguments::LeftVectors() const {
    return vectors_left_;
}

Status
CalcDistanceArguments::SetRightVectors(FloatVecFieldDataPtr vectors) {
    return validate_vectors_with_oper(vectors, [this, &vectors]() { this->vectors_right_ = std::move(vectors); });
}

Status
CalcDistanceArguments::SetRightVectors(BinaryVecFieldDataPtr vectors) {
    return validate_vectors_with_oper(vectors, [this, &vectors]() { this->vectors_right_ = std::move(vectors); });
}

Status
CalcDistanceArguments::SetRightVectors(Int64FieldDataPtr ids, std::string collection_name,
                                       const std::vector<std::string>& partition_names) {
    return validate_vectors_with_oper(ids, collection_name, [this, &ids, &collection_name, &partition_names]() {
        this->vectors_right_ = std::move(ids);
        this->collection_right_ = std::move(collection_name);
        this->partitions_right_ = partition_names;
    });
}

Status
CalcDistanceArguments::SetRightVectors(Int64FieldDataPtr ids, std::string collection_name,
                                       std::vector<std::string>&& partition_names) {
    return validate_vectors_with_oper(ids, collection_name, [this, &ids, &collection_name, &partition_names]() {
        this->vectors_right_ = std::move(ids);
        this->collection_right_ = std::move(collection_name);
        this->partitions_right_ = std::move(partition_names);
    });
}

FieldDataPtr
CalcDistanceArguments::RightVectors() const {
    return vectors_right_;
}

Status
CalcDistanceArguments::SetMetricType(std::string metric) {
    std::transform(metric.begin(), metric.end(), metric.begin(), ::toupper);
    static const std::set<std::string> avaiable_types = {"L2", "IP", "HAMMING", "TANIMOTO"};
    if (avaiable_types.find(metric) == avaiable_types.end()) {
        return {StatusCode::INVALID_AGUMENT, "Invalid metric type!"};
    }
    metric_ = std::move(metric);
    return Status::OK();
}

const std::string&
CalcDistanceArguments::MetricType() const {
    return metric_;
}

Status
CalcDistanceArguments::SetDimension(int32_t dim) {
    if (dim <= 0) {
        return {StatusCode::INVALID_AGUMENT, "Dimension must be greater than 0!"};
    }

    dimension_ = dim;
    return Status::OK();
}

int32_t
CalcDistanceArguments::Dimension() const {
    return dimension_;
}

void
CalcDistanceArguments::SetSqrt(bool sqrt_distance) {
    sqrt_ = sqrt_distance;
}

bool
CalcDistanceArguments::Sqrt() const {
    return sqrt_;
}

const std::string&
CalcDistanceArguments::LeftCollection() const {
    return collection_left_;
}

const std::string&
CalcDistanceArguments::RightCollection() const {
    return collection_right_;
}

const std::vector<std::string>&
CalcDistanceArguments::LeftPartitions() const {
    return partitions_left_;
}

const std::vector<std::string>&
CalcDistanceArguments::RightPartitions() const {
    return partitions_right_;
}

Status
CalcDistanceArguments::Validate() const {
    if (nullptr == vectors_left_ || vectors_left_->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "Vectors on the left of operator cannot be empty!"};
    }

    if (nullptr == vectors_right_ || vectors_right_->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "Vectors on the right of operator cannot be empty!"};
    }

    // To calculate distance, vector type must be equal.
    // If user specified id array, the CalcDistance API will get collection schema to verify.
    if (IsVectorType(vectors_left_->Type()) && IsVectorType(vectors_right_->Type())) {
        if (vectors_left_->Type() != vectors_right_->Type()) {
            return {StatusCode::INVALID_AGUMENT, "Vector types of left and right do not equal!"};
        }

        static const std::set<std::string> float_types = {"L2", "IP"};
        if (vectors_left_->Type() == DataType::FLOAT_VECTOR && float_types.find(metric_) == float_types.end()) {
            return {StatusCode::INVALID_AGUMENT, "Invalid metric type for float vectors!"};
        }

        static const std::set<std::string> binary_types = {"HAMMING", "TANIMOTO"};
        if (vectors_left_->Type() == DataType::BINARY_VECTOR && binary_types.find(metric_) == binary_types.end()) {
            return {StatusCode::INVALID_AGUMENT, "Invalid metric type for binary vectors!"};
        }
    }

    return Status::OK();
}

}  // namespace milvus
