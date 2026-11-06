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

#include <vector>

#include "milvus/Export.h"
#include "milvus/types/RowPolicy.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::ListRowPolicies()
 */
class MILVUS_SDK_API ListRowPoliciesResponse {
 public:
    /**
     * @brief Constructor
     */
    ListRowPoliciesResponse() = default;

    /**
     * @brief Get the row policies.
     * @return the row policies.
     */
    const std::vector<RowPolicy>&
    Policies() const;

    /**
     * @brief Set the row policies.
     * @param [in] policies the row policies.
     */
    void
    SetPolicies(std::vector<RowPolicy>&& policies);

 private:
    std::vector<RowPolicy> policies_;
};

}  // namespace milvus
