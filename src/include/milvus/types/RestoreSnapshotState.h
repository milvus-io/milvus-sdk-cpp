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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief State code for restore snapshot jobs.
 */
enum class RestoreSnapshotStateCode {
    /**
     * @brief The restore job state is unknown.
     */
    UNKNOWN = 0,
    /**
     * @brief The restore job is pending.
     */
    PENDING = 1,
    /**
     * @brief The restore job is running.
     */
    EXECUTING = 2,
    /**
     * @brief The restore job completed.
     */
    COMPLETED = 3,
    /**
     * @brief The restore job failed.
     */
    FAILED = 4,
};

}  // namespace milvus

namespace std {
MILVUS_SDK_API std::string
/**
 * @brief Convert a state code to its string name.
 * @param [in] state the state.
 */
to_string(milvus::RestoreSnapshotStateCode state);
}  // namespace std
