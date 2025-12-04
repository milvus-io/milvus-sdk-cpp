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

namespace milvus {

/**
 * @brief Used by MilvusClientV2::GetCompactionState() and GetCompactionPlans()
 */
class GetCompactionRequest {
 public:
    /**
     * @brief Constructor
     */
    GetCompactionRequest() = default;

    /**
     * @brief Get compaction job id which is returned by Compact().
     */
    int64_t
    CompactionID() const;

    /**
     * @brief Set compaction job id which is returned by Compact().
     */
    void
    SetCompactionID(int64_t id);

    /**
     * @brief Set compaction job id which is returned by Compact().
     */
    GetCompactionRequest&
    WithCompactionID(int64_t id);

 private:
    int64_t compaction_id_{0};
};

using GetCompactionStateRequest = GetCompactionRequest;
using GetCompactionPlansRequest = GetCompactionRequest;

}  // namespace milvus
