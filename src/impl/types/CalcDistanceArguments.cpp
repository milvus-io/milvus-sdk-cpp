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

#include <algorithm>
#include <set>
#include <unordered_map>

namespace milvus {

struct CalcDistanceArguments::Impl {
    FieldDataPtr vectors_left_;
    FieldDataPtr vectors_right_;

    std::string metric_ = "L2";
    bool sqrt_ = false;      // only for "L2"
    int32_t dimension_ = 0;  // only for "HAMMING" and "TANIMOTO"

    // only for id array
    std::string collection_left_;
    std::vector<std::string> partitions_left_;
    std::string collection_right_;
    std::vector<std::string> partitions_right_;
};

CalcDistanceArguments::CalcDistanceArguments() : impl_(new Impl) {
}

CalcDistanceArguments::CalcDistanceArguments(CalcDistanceArguments&&) noexcept = default;

CalcDistanceArguments::~CalcDistanceArguments() = default;

Status
CalcDistanceArguments::SetLeftVectors(const FloatVecFieldDataPtr& vectors) {
    if (nullptr == vectors || vectors->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
    }

    impl_->vectors_left_ = vectors;
    return Status::OK();
}

Status
CalcDistanceArguments::SetLeftVectors(const BinaryVecFieldDataPtr& vectors) {
    if (nullptr == vectors || vectors->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
    }

    impl_->vectors_left_ = vectors;
    return Status::OK();
}

Status
CalcDistanceArguments::SetLeftVectors(const Int64FieldDataPtr& ids, const std::string& collection_name,
                                      const std::vector<std::string>& partition_names) {
    if (nullptr == ids || ids->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
    }

    if (ids->Name().empty()) {
        return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
    }

    if (collection_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Collection name cannot be empty!"};
    }

    impl_->vectors_left_ = ids;
    impl_->collection_left_ = collection_name;
    impl_->partitions_left_ = partition_names;
    return Status::OK();
}

FieldDataPtr
CalcDistanceArguments::LeftVectors() const {
    return impl_->vectors_left_;
}

Status
CalcDistanceArguments::SetRightVectors(const FloatVecFieldDataPtr& vectors) {
    if (nullptr == vectors || vectors->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
    }

    impl_->vectors_right_ = vectors;
    return Status::OK();
}

Status
CalcDistanceArguments::SetRightVectors(const BinaryVecFieldDataPtr& vectors) {
    if (nullptr == vectors || vectors->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
    }

    impl_->vectors_right_ = vectors;
    return Status::OK();
}

Status
CalcDistanceArguments::SetRightVectors(const Int64FieldDataPtr& ids, const std::string& collection_name,
                                       const std::vector<std::string>& partition_names) {
    if (nullptr == ids || ids->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
    }

    if (ids->Name().empty()) {
        return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
    }

    if (collection_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Collection name cannot be empty!"};
    }

    impl_->vectors_right_ = ids;
    impl_->collection_right_ = collection_name;
    impl_->partitions_right_ = partition_names;
    return Status::OK();
}

FieldDataPtr
CalcDistanceArguments::RightVectors() const {
    return impl_->vectors_right_;
}

Status
CalcDistanceArguments::SetMetricType(const std::string& metric) {
    std::string upper_metric = metric;
    std::transform(upper_metric.begin(), upper_metric.end(), upper_metric.begin(), ::toupper);
    static const std::set<std::string> avaiable_types = {"L2", "IP", "HAMMING", "TANIMOTO"};
    if (avaiable_types.find(upper_metric) == avaiable_types.end()) {
        return {StatusCode::INVALID_AGUMENT, "Invalid metric type!"};
    }
    impl_->metric_ = std::move(upper_metric);
    return Status::OK();
}

const std::string&
CalcDistanceArguments::MetricType() const {
    return impl_->metric_;
}

Status
CalcDistanceArguments::SetDimension(int32_t dim) {
    if (dim <= 0) {
        return {StatusCode::INVALID_AGUMENT, "Dimension must be greater than 0!"};
    }

    impl_->dimension_ = dim;
    return Status::OK();
}

int32_t
CalcDistanceArguments::Dimension() const {
    return impl_->dimension_;
}

void
CalcDistanceArguments::SetSqrt(bool sqrt_distance) {
    impl_->sqrt_ = sqrt_distance;
}

bool
CalcDistanceArguments::Sqrt() const {
    return impl_->sqrt_;
}

const std::string&
CalcDistanceArguments::LeftCollection() const {
    return impl_->collection_left_;
}

const std::string&
CalcDistanceArguments::RightCollection() const {
    return impl_->collection_right_;
}

const std::vector<std::string>&
CalcDistanceArguments::LeftPartitions() const {
    return impl_->partitions_left_;
}

const std::vector<std::string>&
CalcDistanceArguments::RightPartitions() const {
    return impl_->partitions_right_;
}

Status
CalcDistanceArguments::Validate() const {
    if (nullptr == impl_->vectors_left_ || impl_->vectors_left_->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "Vectors on the left of operator cannot be empty!"};
    }

    if (nullptr == impl_->vectors_right_ || impl_->vectors_right_->Count() == 0) {
        return {StatusCode::INVALID_AGUMENT, "Vectors on the right of operator cannot be empty!"};
    }

    // To calculate distance, vector type must be equal.
    // If user specified id array, the CalcDistance API will get collection schema to verify.
    if (IsVectorType(impl_->vectors_left_->Type()) && IsVectorType(impl_->vectors_right_->Type())) {
        if (impl_->vectors_left_->Type() != impl_->vectors_right_->Type()) {
            return {StatusCode::INVALID_AGUMENT, "Vector types of left and right do not equal!"};
        }

        static const std::set<std::string> float_types = {"L2", "IP"};
        if (impl_->vectors_left_->Type() == DataType::FLOAT_VECTOR &&
            float_types.find(impl_->metric_) == float_types.end()) {
            return {StatusCode::INVALID_AGUMENT, "Invalid metric type for float vectors!"};
        }

        static const std::set<std::string> binary_types = {"HAMMING", "TANIMOTO"};
        if (impl_->vectors_left_->Type() == DataType::BINARY_VECTOR &&
            binary_types.find(impl_->metric_) == binary_types.end()) {
            return {StatusCode::INVALID_AGUMENT, "Invalid metric type for binary vectors!"};
        }
    }

    return Status::OK();
}

}  // namespace milvus
