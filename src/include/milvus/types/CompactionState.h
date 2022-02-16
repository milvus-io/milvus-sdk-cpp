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
 * @brief State Code for compaction
 */
enum class CompactionStateCode {
    UNKNOWN = 0,
    EXECUTING = 1,
    COMPLETED = 2,
};

/**
 * @brief Compaction state. Used by MilvusClient::GetCompactionState().
 */
class CompactionState {
 public:
    CompactionState() = default;

    /**
     * @brief Compaction state code.
     */
    CompactionStateCode
    State() const {
        return state_code_;
    }

    /**
     * @brief Set compaction state code.
     */
    void
    SetState(const CompactionStateCode& state) {
        state_code_ = state;
    }

    /**
     * @brief The executing plan id.
     */
    int64_t
    ExecutingPlan() const {
        return executing_plan_;
    }

    /**
     * @brief Set the executing plan id.
     */
    void
    SetExecutingPlan(const int64_t id) {
        executing_plan_ = id;
    }

    /**
     * @brief The timeout plan id.
     */
    int64_t
    TimeoutPlan() const {
        return timeout_plan_;
    }

    /**
     * @brief Set the timeout plan id.
     */
    void
    SetTimeoutPlan(const int64_t id) {
        timeout_plan_ = id;
    }

    /**
     * @brief The completed plan id.
     */
    int64_t
    CompletedPlan() const {
        return completed_plan_;
    }

    /**
     * @brief Set the completed plan id.
     */
    void
    SetCompletedPlan(const int64_t id) {
        completed_plan_ = id;
    }

 private:
    CompactionStateCode state_code_{CompactionStateCode::UNKNOWN};

    int64_t executing_plan_ = 0;
    int64_t timeout_plan_ = 0;
    int64_t completed_plan_ = 0;
};

}  // namespace milvus
