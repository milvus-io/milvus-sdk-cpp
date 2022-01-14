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

#pragma once

#include <algorithm>
#include <set>
#include <unordered_map>

#include "../Status.h"
#include "Constants.h"
#include "FieldData.h"

namespace milvus {

/**
 * @brief Arguments for CalcDistance().
 */
class CalcDistanceArguments {
 public:
    CalcDistanceArguments() = default;

    /**
     * @brief Set the float vectors on the left of operator, without field name.
     */
    Status
    SetLeftVectors(FloatVecFieldDataPtr&& vectors) {
        if (nullptr == vectors || vectors->Count() == 0) {
            return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
        }

        vectors_left_ = std::move(vectors);
        return Status::OK();
    }

    /**
     * @brief Set the binary vectors on the left of operator, without field name.
     */
    Status
    SetLeftVectors(BinaryVecFieldDataPtr&& vectors) {
        if (nullptr == vectors || vectors->Count() == 0) {
            return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
        }

        vectors_left_ = std::move(vectors);
        return Status::OK();
    }

    /**
     * @brief Set id array of the vectors on the left of operator, must specify field name and collection name.
     * Partition names is optinal, to narrow down the query scope.
     */
    Status
    SetLeftVectors(Int64FieldDataPtr&& ids, const std::string& collection_name,
                   std::vector<std::string>&& partition_names = {}) {
        if (nullptr == ids || ids->Count() == 0) {
            return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
        }

        if (ids->Name().empty()) {
            return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
        }

        if (collection_name.empty()) {
            return {StatusCode::INVALID_AGUMENT, "Collection name cannot be empty!"};
        }

        vectors_left_ = std::move(ids);
        collection_left_ = collection_name;
        partitions_left_ = std::move(partition_names);
        return Status::OK();
    }

    /**
     * @brief Get the vectors on the left of operator.
     */
    FieldDataPtr
    LeftVectors() const {
        return std::static_pointer_cast<Field>(vectors_left_);
    }

    /**
     * @brief Set the float vectors on the right of operator, without field name.
     */
    Status
    SetRightVectors(FloatVecFieldDataPtr&& vectors) {
        if (nullptr == vectors || vectors->Count() == 0) {
            return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
        }

        vectors_right_ = std::move(vectors);
        return Status::OK();
    }

    /**
     * @brief Set the binary vectors on the right of operator, without field name.
     */
    Status
    SetRightVectors(BinaryVecFieldDataPtr&& vectors) {
        if (nullptr == vectors || vectors->Count() == 0) {
            return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
        }

        vectors_right_ = std::move(vectors);
        return Status::OK();
    }

    /**
     * @brief Set id array of the vectors on the right of operator, must specify field name and collection name.
     * Partition names is optinal, to narrow down the query scope.
     */
    Status
    SetRightVectors(Int64FieldDataPtr&& ids, const std::string& collection_name,
                    std::vector<std::string>&& partition_names = {}) {
        if (nullptr == ids || ids->Count() == 0) {
            return {StatusCode::INVALID_AGUMENT, "Input vectors cannot be empty!"};
        }

        if (ids->Name().empty()) {
            return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
        }

        if (collection_name.empty()) {
            return {StatusCode::INVALID_AGUMENT, "Collection name cannot be empty!"};
        }

        vectors_right_ = std::move(ids);
        collection_right_ = collection_name;
        partitions_right_ = std::move(partition_names);
        return Status::OK();
    }

    /**
     * @brief Get the vectors on the right of operator.
     */
    FieldDataPtr
    RightVectors() const {
        return std::static_pointer_cast<Field>(vectors_right_);
    }

    /**
     * @brief Set metric type of calculation, options: "L2"/"IP"/"HAMMING"/"TANIMOTO", default is "L2". The type string
     * is case insensitive. "L2" and "IP" is only for float vectors, "HAMMING" and "TANIMOTO" is for binary vectors.
     */
    Status
    SetMetricType(std::string&& metric) {
        std::string upper_metric = std::move(metric);
        std::transform(upper_metric.begin(), upper_metric.end(), upper_metric.begin(), ::toupper);
        static const std::set<std::string> avaiable_types = {"L2", "IP", "HAMMING", "TANIMOTO"};
        if (avaiable_types.find(upper_metric) == avaiable_types.end()) {
            return {StatusCode::INVALID_AGUMENT, "Invalid metric type!"};
        }

        metric_ = std::move(upper_metric);
        return Status::OK();
    }

    /**
     * @brief Get the specified metric type.
     */
    std::string
    MetricType() const {
        return metric_;
    }

    /**
     * @brief Specify dimension value if dimension is not a multiple of 8, otherwise the dimension will be calculted by
     * vector data length, only for "HAMMING" and "TANIMOTO".
     */
    Status
    SetDimension(int32_t dim) {
        if (dim <= 0) {
            return {StatusCode::INVALID_AGUMENT, "Dimension must be greater than 0!"};
        }

        dimension_ = dim;
        return Status::OK();
    }

    /**
     * @brief Get specified dimension, only for "HAMMING" and "TANIMOTO".
     */
    int32_t
    Dimension() const {
        return dimension_;
    }

    /**
     * @brief Calculate extraction of a root of distance values, default is false, only for "L2" metric type.
     */
    void
    SetSqrt(bool sqrt_distance) {
        sqrt_ = sqrt_distance;
    }

    /**
     * @brief Get flag of sqrt, only for "L2" metric type.
     */
    bool
    Sqrt() const {
        return sqrt_;
    }

    /**
     * @brief Get the collection which left vectors belong. Only for vector id array.
     */
    std::string
    LeftCollection() const {
        return collection_left_;
    }

    /**
     * @brief Get the collection which right vectors belong. Only for vector id array.
     */
    std::string
    RightCollection() const {
        return collection_right_;
    }

    /**
     * @brief Get the partitions which left vectors belong. Only for vector id array.
     */
    const std::vector<std::string>&
    LeftPartitions() const {
        return partitions_left_;
    }

    /**
     * @brief Get the partitions which right vectors belong. Only for vector id array.
     */
    const std::vector<std::string>&
    RightPartitions() const {
        return partitions_right_;
    }

    /**
     * @brief Basic validation for the input arguments.
     */
    Status
    Validate() const {
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

 private:
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

}  // namespace milvus
