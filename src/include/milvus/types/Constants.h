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

// Windows DLL consumers need dllimport for exported constant data symbols.
#if defined(_WIN32) && defined(MILVUS_SDK_SHARED)
#if defined(MILVUS_SDK_BUILDING_LIBRARY)
#define MILVUS_SDK_API __declspec(dllexport)
#else
#define MILVUS_SDK_API __declspec(dllimport)
#endif
#else
#define MILVUS_SDK_API
#endif

namespace milvus {

// const names for external common usage
extern MILVUS_SDK_API const char* INDEX_TYPE;
extern MILVUS_SDK_API const char* METRIC_TYPE;
extern MILVUS_SDK_API const char* NLIST;
extern MILVUS_SDK_API const char* NPROBE;
extern MILVUS_SDK_API const char* DIM;
extern MILVUS_SDK_API const char* MAX_LENGTH;
extern MILVUS_SDK_API const char* MAX_CAPACITY;
extern MILVUS_SDK_API const char* DYNAMIC_FIELD;
extern MILVUS_SDK_API const char* SPARSE_INDICES;
extern MILVUS_SDK_API const char* SPARSE_VALUES;
extern MILVUS_SDK_API const char* MMAP_ENABLED;
extern MILVUS_SDK_API const char* COLLECTION_TTL_SECONDS;
extern MILVUS_SDK_API const char* COLLECTION_REPLICA_NUMBER;
extern MILVUS_SDK_API const char* COLLECTION_RESOURCE_GROUPS;
extern MILVUS_SDK_API const char* DATABASE_REPLICA_NUMBER;
extern MILVUS_SDK_API const char* DATABASE_RESOURCE_GROUPS;

/////////////////////////////////////////////////////////////////////////////////
// the following methods are reserved to compatible with old client code
/**
 * @brief Global definition for row count label.
 */
inline std::string
KeyRowCount() {
    return "row_count";
}

/**
 * @brief Global definition for index type label.
 */
inline std::string
KeyIndexType() {
    return INDEX_TYPE;
}

/**
 * @brief Global definition for metric type label.
 */
inline std::string
KeyMetricType() {
    return METRIC_TYPE;
}

/**
 * @brief Global definition for metric type label.
 */
inline std::string
KeyParams() {
    return "params";
}

/**
 * @brief Global definition for vector dimension label.
 */
inline std::string
FieldDim() {
    return DIM;
}

/**
 * @brief Max length field name for varchar field.
 */
inline std::string
FieldMaxLength() {
    return MAX_LENGTH;
}

/**
 * @brief Global definition for strong guarantee timestamp.
 */
inline uint64_t
GuaranteeStrongTs() {
    return 0;
}

/**
 * @brief Global definition for eventually guarantee timestamp.
 */
inline uint64_t
GuaranteeEventuallyTs() {
    return 1;
}

}  // namespace milvus
