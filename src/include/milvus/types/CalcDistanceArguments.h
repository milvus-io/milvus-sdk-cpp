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

#include <vector>

#include "../Status.h"
#include "FieldData.h"

namespace milvus {

/**
 * @brief Arguments for MilvusClient::CalcDistance().
 */
class CalcDistanceArguments {
 public:
    /**
     * @brief Constructor
     */
    CalcDistanceArguments();

    /**
     * @brief Move constructor
     */
    CalcDistanceArguments(CalcDistanceArguments&&) noexcept;

    /**
     * @brief Destructor
     */
    ~CalcDistanceArguments();

    /**
     * @brief Set the float vectors on the left of operator, without field name.
     */
    Status
    SetLeftVectors(const FloatVecFieldDataPtr& vectors);

    /**
     * @brief Set the binary vectors on the left of operator, without field name.
     */
    Status
    SetLeftVectors(const BinaryVecFieldDataPtr& vectors);

    /**
     * @brief Set id array of the vectors on the left of operator, must specify field name and collection name.
     * Partition names is optinal, to narrow down the query scope.
     */
    Status
    SetLeftVectors(const Int64FieldDataPtr& ids, const std::string& collection_name,
                   const std::vector<std::string>& partition_names = {});

    /**
     * @brief Get the vectors on the left of operator.
     */
    FieldDataPtr
    LeftVectors() const;

    /**
     * @brief Set the float vectors on the right of operator, without field name.
     */
    Status
    SetRightVectors(const FloatVecFieldDataPtr& vectors);

    /**
     * @brief Set the binary vectors on the right of operator, without field name.
     */
    Status
    SetRightVectors(const BinaryVecFieldDataPtr& vectors);

    /**
     * @brief Set id array of the vectors on the right of operator, must specify field name and collection name.
     * Partition names is optinal, to narrow down the query scope.
     */
    Status
    SetRightVectors(const Int64FieldDataPtr& ids, const std::string& collection_name,
                    const std::vector<std::string>& partition_names = {});

    /**
     * @brief Get the vectors on the right of operator.
     */
    FieldDataPtr
    RightVectors() const;

    /**
     * @brief Set metric type of calculation, options: "L2"/"IP"/"HAMMING"/"TANIMOTO", default is "L2". The type string
     * is case insensitive. "L2" and "IP" is only for float vectors, "HAMMING" and "TANIMOTO" is for binary vectors.
     */
    Status
    SetMetricType(const std::string& metric);

    /**
     * @brief Get the specified metric type.
     */
    const std::string&
    MetricType() const;

    /**
     * @brief Specify dimension value if dimension is not a multiple of 8, otherwise the dimension will be calculted by
     * vector data length, only for "HAMMING" and "TANIMOTO".
     */
    Status
    SetDimension(int32_t dim);

    /**
     * @brief Get specified dimension, only for "HAMMING" and "TANIMOTO".
     */
    int32_t
    Dimension() const;

    /**
     * @brief Calculate extraction of a root of distance values, default is false, only for "L2" metric type.
     */
    void
    SetSqrt(bool sqrt_distance);

    /**
     * @brief Get flag of sqrt, only for "L2" metric type.
     */
    bool
    Sqrt() const;

    /**
     * @brief Get the collection which left vectors belong. Only for vector id array.
     */
    const std::string&
    LeftCollection() const;

    /**
     * @brief Get the collection which right vectors belong. Only for vector id array.
     */
    const std::string&
    RightCollection() const;

    /**
     * @brief Get the partitions which left vectors belong. Only for vector id array.
     */
    const std::vector<std::string>&
    LeftPartitions() const;

    /**
     * @brief Get the partitions which right vectors belong. Only for vector id array.
     */
    const std::vector<std::string>&
    RightPartitions() const;

    /**
     * @brief Basic validation for the input arguments.
     */
    Status
    Validate() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace milvus
