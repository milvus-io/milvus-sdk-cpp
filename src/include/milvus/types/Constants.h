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
#include <string>

namespace milvus {

/**
 * @brief Global definition for row count label
 */
inline const std::string
KeyRowCount() {
    return "row_count";
}

/**
 * @brief Global definition for index type label
 */
inline const std::string
KeyIndexType() {
    return "index_type";
}

/**
 * @brief Global definition for metric type label
 */
inline const std::string
KeyMetricType() {
    return "metric_type";
}

/**
 * @brief Global definition for metric type label
 */
inline const std::string
KeyParams() {
    return "params";
}

/**
 * @brief Global definition for vector dimension label
 */
inline const std::string
FieldDim() {
    return "dim";
}

/**
 * @brief Global definition for strong guarantee timestamp
 */
inline uint64_t
GuaranteeStrongTs() {
    return 0;
}

/**
 * @brief Global definition for eventually guarantee timestamp
 */
inline uint64_t
GuaranteeEventuallyTs() {
    return 1;
}

/**
 * @brief The logical bits in hybird timestamp
 */
constexpr size_t inline HybirdtsLogicalBits() {
    return 18;
}

/**
 * @brief The logical bits mask for hybird timestamp
 */
constexpr uint64_t inline HybirdtsLogicalBitsMask() {
    return (1 << HybirdtsLogicalBits()) - 1;
}

}  // namespace milvus
