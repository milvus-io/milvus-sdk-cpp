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
 * @brief Type of a row policy.
 *
 * The numeric values MUST match the corresponding values of
 * proto::milvus::RowPolicyType in milvus-proto (see milvus.proto).
 * Do not renumber existing values.
 */
enum class RowPolicyType {
    /**
     * @brief Unknown or unset row policy type.
     */
    UNKNOWN = 0,
    /**
     * @brief Permissive policy; at least one applicable permissive policy is
     * required to admit a row.
     */
    PERMISSIVE = 1,
    /**
     * @brief Restrictive policy; every applicable restrictive policy must admit
     * a row.
     */
    RESTRICTIVE = 2,
};

}  // namespace milvus
