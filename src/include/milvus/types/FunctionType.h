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
 * @brief Supported function types.
 * Numeric values mirror schema.proto and are part of the client/server protocol contract.
 * Existing values must remain stable for wire compatibility.
 */
enum class FunctionType {
    /**
     * @brief Unknown function type.
     */
    UNKNOWN = 0,
    /**
     * @brief BM25 text scoring function.
     */
    BM25 = 1,
    /**
     * @brief Text embedding function.
     */
    TEXTEMBEDDING = 2,
    /**
     * @brief Rerank function.
     */
    RERANK = 3,
    /**
     * @brief MinHash signature function for binary vectors.
     */
    MINHASH = 4,
    /**
     * @brief Molecular fingerprint function.
     */
    MOLFINGERPRINT = 5,
};
}  // namespace milvus

namespace std {
MILVUS_SDK_API std::string to_string(milvus::FunctionType);
}  // namespace std
