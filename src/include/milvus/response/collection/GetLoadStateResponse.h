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

#include "../../types/LoadState.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::GetLoadState()
 */
class GetLoadStateResponse {
 public:
    /**
     * @brief Constructor
     */
    GetLoadStateResponse() = default;

    // Getter and Setter for state_
    LoadState
    State() const;
    void
    SetState(LoadState state);

    // Getter and Setter for progress_
    int64_t
    Progress() const;
    void
    SetProgress(int64_t progress);

 private:
    LoadState state_{LoadState::LOAD_STATE_NOT_EXIST};
    int64_t progress_{0};
};

}  // namespace milvus
