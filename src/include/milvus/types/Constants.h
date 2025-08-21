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

// const names for external common usage
extern const char* INDEX_TYPE;
extern const char* METRIC_TYPE;
extern const char* NLIST;
extern const char* NPROBE;
extern const char* DIM;
extern const char* MAX_LENGTH;
extern const char* MAX_CAPACITY;
extern const char* DYNAMIC_FIELD;
extern const char* SPARSE_INDICES;
extern const char* SPARSE_VALUES;
extern const char* MMAP_ENABLED;
extern const char* COLLECTION_TTL_SECONDS;
extern const char* COLLECTION_REPLICA_NUMBER;
extern const char* COLLECTION_RESOURCE_GROUPS;
extern const char* DATABASE_REPLICA_NUMBER;
extern const char* DATABASE_RESOURCE_GROUPS;

/////////////////////////////////////////////////////////////////////////////////
// the following methods are reserved to compatible with old client code
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
    return INDEX_TYPE;
}

/**
 * @brief Global definition for metric type label
 */
inline std::string
KeyMetricType() {
    return METRIC_TYPE;
}

/**
 * @brief Global definition for metric type label
 */
inline std::string
KeyParams() {
    return "params";
}

/**
 * @brief Global definition for vector dimension label
 */
inline std::string
FieldDim() {
    return DIM;
}

/**
 * @brief Max length field name for varchar field
 */
inline std::string
FieldMaxLength() {
    return MAX_LENGTH;
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

}  // namespace milvus
