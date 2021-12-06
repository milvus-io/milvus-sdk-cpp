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
 * @brief State Code for index
 */
enum class IndexStateCode {
    NONE = 0,
    UNISSUED = 1,
    IN_PROGRESS = 2,
    FINISHED = 3,
    FAILED = 4,
};

/**
 * @brief Index state. Used by GetIndexState().
 */
class IndexState {
 public:
    /**
     * @brief Index state code.
     */
    IndexStateCode
    StateCode() const {
        return state_code_;
    }

    /**
     * @brief Failed reason why the index failed to build.
     */
    std::string
    FailedReason() const {
        return failed_reason_;
    }

 private:
    IndexStateCode state_code_{IndexStateCode::NONE};
    std::string failed_reason_;
};

/**
 * @brief Index progress. Used by GetIndexBuildProgress().
 */
class IndexProgress {
 public:
    /**
     * @brief Row count already indexed.
     */
    int64_t
    IndexedRows() const {
        return indexed_rows_;
    }

    /**
     * @brief Total rows need to be indxed.
     */
    int64_t
    TotalRows() const {
        return total_rows_;
    }

 private:
    int64_t indexed_rows_ = 0;
    int64_t total_rows_ = 0;
};

}  // namespace milvus
