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

#include "../../types/QueryResults.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Query()
 */
class MILVUS_SDK_API QueryResponse {
 public:
    /**
     * @brief Constructor
     */
    QueryResponse() = default;

    /**
     * @brief Get result of query operation.
     */
    const QueryResults&
    Results() const;

    /**
     * @brief Set result of query operation.
     */
    void
    SetResults(QueryResults&& results);

    uint64_t
    SessionTs() const;

    void
    SetSessionTs(uint64_t session_ts);

 private:
    QueryResults results_;
    uint64_t session_ts_{0};
};

using GetResponse = QueryResponse;

}  // namespace milvus
