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

#include "milvus/types/CompactionState.h"

namespace milvus {

CompactionState::CompactionState() = default;

CompactionStateCode
CompactionState::State() const {
    return state_code_;
}

void
CompactionState::SetState(CompactionStateCode state) {
    state_code_ = state;
}

int64_t
CompactionState::ExecutingPlan() const {
    return executing_plan_;
}

void
CompactionState::SetExecutingPlan(int64_t id) {
    executing_plan_ = id;
}

int64_t
CompactionState::TimeoutPlan() const {
    return timeout_plan_;
}

void
CompactionState::SetTimeoutPlan(int64_t id) {
    timeout_plan_ = id;
}

int64_t
CompactionState::CompletedPlan() const {
    return completed_plan_;
}

void
CompactionState::SetCompletedPlan(int64_t id) {
    completed_plan_ = id;
}

}  // namespace milvus
