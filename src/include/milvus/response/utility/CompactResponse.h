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
 * @brief Used by MilvusClientV2::Compact()
 */
class CompactResponse {
 public:
    /**
     * @brief Constructor
     */
    CompactResponse() = default;

    // Getter and Setter for compaction_id_
    int64_t
    CompactionID() const;

    void
    SetCompactionID(int64_t id);

    // Getter and Setter for compaction_plan_count_
    int64_t
    CompactionPlanCount() const;

    void
    SetCompactionPlanCount(int64_t id);

 private:
    int64_t compaction_id_{0};
    int64_t compaction_plan_count_{0};
};

}  // namespace milvus
