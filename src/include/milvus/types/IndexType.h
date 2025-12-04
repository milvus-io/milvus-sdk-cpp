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
 * @brief Supported index types.
 */
enum class IndexType {
    INVALID = 0,

    // CPU indexes only for float vectors
    // the value of each index type doesn't matter, the sdk converts IndexType to string name and
    // passes the name to the server
    // Note: in milvus 2.4+, IVF_HNSW/RHNSW_FLAT/RHNSW_SQ/RHNSW_PQ/ANNOY have been deprecated
    FLAT = 1,
    IVF_FLAT = 2,
    IVF_SQ8 = 3,
    IVF_PQ = 4,
    HNSW = 5,
    DISKANN = 6,
    AUTOINDEX = 7,
    SCANN = 8,
    HNSW_SQ = 9,
    HNSW_PQ = 10,
    HNSW_PRQ = 11,
    IVF_RABITQ = 12,

    // GPU indexes only for float vectors
    GPU_IVF_FLAT = 201,
    GPU_IVF_PQ = 202,
    GPU_BRUTE_FORCE = 203,
    GPU_CAGRA = 204,

    // Indexes for binary vectors
    BIN_FLAT = 1001,
    BIN_IVF_FLAT = 1002,
    MINHASH_LSH = 1003,

    // Only for varchar type field
    TRIE = 1101,
    // Only for scalar type field
    STL_SORT = 1102,  // only for numeric type field
    INVERTED = 1103,  // works for all scalar fields except JSON type field
    BITMAP = 1104,    // works for all scalar fields except JSON, FLOAT and DOUBLE type fields

    // Only for varchar type field and json_path of JSON field
    NGRAM = 1105,

    // Only for sparse vectors
    SPARSE_INVERTED_INDEX = 1201,
    SPARSE_WAND = 1202
};
}  // namespace milvus

namespace std {
std::string to_string(milvus::IndexType);
}  // namespace std
