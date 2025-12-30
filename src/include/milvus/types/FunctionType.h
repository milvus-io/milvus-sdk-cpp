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
 * @brief Supported function types.
 * Note: in v2.4, only support RERANK for hybrid search
 * in v2.5, we have BM25 = 1
 * in v2.6, we have TEXTEMBEDDING = 2
 */
enum class FunctionType {
    UNKNOWN = 0,
    BM25 = 1,
    RERANK = 3,
};
}  // namespace milvus

namespace std {
std::string to_string(milvus::FunctionType);
}  // namespace std
