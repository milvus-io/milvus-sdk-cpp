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
 * @brief Supported metric types.
 */
enum class MetricType {
    INVALID = 0,  // deprecated, replaced by DEFAULT
    DEFAULT = 0,  // the server automatically determines metric type
    L2 = 1,
    IP = 2,
    COSINE = 3,

    // The following values are for binary vectors
    HAMMING = 101,
    JACCARD = 102,
    MHJACCARD = 103,

    BM25 = 201,  // Only for sparse vector with BM25

    // Only for float vector inside struct
    MAX_SIM_COSINE = 301,
    MAX_SIM_IP = 302,
    MAX_SIM_L2 = 303,

    // Only for binary vector inside struct
    MAX_SIM_JACCARD = 401,
    MAX_SIM_HAMMING = 402,

    // Note: in milvus 2.4+, TANIMOTO/SUBSTRUCTURE/SUPERSTRUCTURE are no longer supported
};
}  // namespace milvus

namespace std {
std::string to_string(milvus::MetricType);
}  // namespace std
