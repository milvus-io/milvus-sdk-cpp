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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Data type of field
 *
 * The numeric values MUST match the corresponding values of
 * proto::schema::DataType in milvus-proto (see schema.proto), so that
 * DataTypeCast() can map between them. Do not renumber existing values or
 * reuse a slot that is already taken; add new types with the matching proto
 * value. Existing values are part of the public API and must stay stable.
 */
enum class DataType {
    /**
     * @brief Unknown data type.
     */
    UNKNOWN = 0,

    /**
     * @brief Boolean scalar field.
     */
    BOOL = 1,
    /**
     * @brief 8-bit signed integer scalar field.
     */
    INT8 = 2,
    /**
     * @brief 16-bit signed integer scalar field.
     */
    INT16 = 3,
    /**
     * @brief 32-bit signed integer scalar field.
     */
    INT32 = 4,
    /**
     * @brief 64-bit signed integer scalar field.
     */
    INT64 = 5,

    /**
     * @brief 32-bit floating-point scalar field.
     */
    FLOAT = 10,
    /**
     * @brief 64-bit floating-point scalar field.
     */
    DOUBLE = 11,

    // STRING not available
    // STRING = 20,
    /**
     * @brief Variable-length string scalar field.
     */
    VARCHAR = 21,

    /**
     * @brief Array field of a fixed element type.
     */
    ARRAY = 22,
    /**
     * @brief JSON object scalar field.
     */
    JSON = 23,
    /**
     * @brief Geometry (WKT/WKB) scalar field.
     */
    GEOMETRY = 24,
    /**
     * @brief Full-text searchable text scalar field.
     */
    TEXT = 25,
    /**
     * @brief Time-zone-aware timestamp scalar field.
     */
    TIMESTAMPTZ = 26,

    /**
     * @brief Binary vector field.
     */
    BINARY_VECTOR = 100,
    /**
     * @brief Float vector field.
     */
    FLOAT_VECTOR = 101,
    /**
     * @brief Float16 vector field.
     */
    FLOAT16_VECTOR = 102,
    /**
     * @brief BFloat16 vector field.
     */
    BFLOAT16_VECTOR = 103,
    /**
     * @brief Sparse float vector field.
     */
    SPARSE_FLOAT_VECTOR = 104,
    /**
     * @brief Int8 vector field.
     */
    INT8_VECTOR = 105,

    /**
     * @brief Struct field holding nested sub-fields.
     */
    STRUCT = 201,
};

}  // namespace milvus

namespace std {
MILVUS_SDK_API std::string to_string(milvus::DataType);
}  // namespace std
