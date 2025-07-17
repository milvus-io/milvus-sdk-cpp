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
#include <cstdint>
#include <string>

namespace milvus {

/**
 * @brief Global definition for row count label
 */
inline std::string
KeyRowCount() {
    return "row_count";
}

/**
 * @brief Global definition for index type label
 */
inline std::string
KeyIndexType() {
    return "index_type";
}

/**
 * @brief Global definition for metric type label
 */
inline std::string
KeyMetricType() {
    return "metric_type";
}

/**
 * @brief Global definition for round decimal of search results
 */
inline std::string
KeyRoundDecimal() {
    return "round_decimal";
}

/**
 * @brief Index parameter for IVF
 */
inline const std::string&
KeyNlist() {
    static std::string nlist = "nlist";
    return nlist;
}

/**
 * @brief Index parameter for IVF
 */
inline const std::string&
KeyNprobe() {
    static std::string nprobe = "nprobe";
    return nprobe;
}

/**
 * @brief Global definition for vector dimension label
 */
inline std::string
FieldDim() {
    return "dim";
}

/**
 * @brief Max length field name for varchar field
 */
inline std::string
FieldMaxLength() {
    return "max_length";
}

/**
 * @brief Max capacity field name for array field
 */
inline std::string
FieldMaxCapacity() {
    return "max_capacity";
}

inline std::string
DynamicFieldName() {
    return "$meta";
}

}  // namespace milvus
