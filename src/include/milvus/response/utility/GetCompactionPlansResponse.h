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

#include "milvus/Export.h"
#include "milvus/types/CompactionPlan.h"
#include "milvus/types/CompactionState.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::GetCompactionPlans()
 */
class MILVUS_SDK_API GetCompactionPlansResponse {
 public:
    /**
     * @brief Constructor
     */
    GetCompactionPlansResponse() = default;

    /**
     * @brief Get plans of the compaction.
     */
    const CompactionPlans&
    Plans() const;

    /**
     * @brief Set plans of the compaction.
     */
    void
    SetPlans(CompactionPlans&& plans);

    /**
     * @brief Get the id of the compaction.
     */
    int64_t
    CompactionID() const;

    /**
     * @brief Set the id of the compaction.
     */
    void
    SetCompactionID(int64_t compaction_id);

    /**
     * @brief Get the state of the compaction.
     */
    CompactionStateCode
    State() const;

    /**
     * @brief Set the state of the compaction.
     */
    void
    SetState(CompactionStateCode state);

 private:
    CompactionPlans plans_;
    int64_t compaction_id_{0};
    CompactionStateCode state_{CompactionStateCode::UNKNOWN};
};

}  // namespace milvus
