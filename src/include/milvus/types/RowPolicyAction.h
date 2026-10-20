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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Row-bearing operation covered by a row policy.
 *
 * The numeric values MUST match the corresponding values of
 * proto::milvus::RowPolicyAction in milvus-proto (see milvus.proto).
 * Do not renumber existing values.
 */
enum class RowPolicyAction {
    /**
     * @brief Query rows.
     */
    QUERY = 0,
    /**
     * @brief Search rows.
     */
    SEARCH = 1,
    /**
     * @brief Insert rows.
     */
    INSERT = 2,
    /**
     * @brief Delete rows.
     * Named DELETE_ROWS because the Windows SDK defines a `DELETE` macro that
     * would otherwise corrupt the enumerator.
     */
    DELETE_ROWS = 3,
    /**
     * @brief Upsert rows.
     */
    UPSERT = 4,
    /**
     * @brief Query through an iterator.
     */
    QUERY_ITERATOR = 5,
    /**
     * @brief Search through an iterator.
     */
    SEARCH_ITERATOR = 6,
    /**
     * @brief Hybrid search.
     */
    HYBRID_SEARCH = 7,
};

}  // namespace milvus
