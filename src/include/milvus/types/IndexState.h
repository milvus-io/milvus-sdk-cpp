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
#include <string>

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief State Code for index
 */
enum class IndexStateCode {
    /**
     * @brief No index state has been reported.
     */
    NONE = 0,
    /**
     * @brief The index build task has not been issued.
     */
    UNISSUED = 1,
    /**
     * @brief The index is being built.
     */
    IN_PROGRESS = 2,
    /**
     * @brief The index has been built successfully.
     */
    FINISHED = 3,
    /**
     * @brief The index build failed.
     */
    FAILED = 4,
    /**
     * @brief The index build is being retried.
     */
    RETRY = 5,
};

/**
 * @brief Index state. Used by MilvusClient::GetIndexState().
 */
class MILVUS_SDK_API IndexState {
 public:
    /**
     * @brief Index state code.
     * @return the state code.
     */
    IndexStateCode
    StateCode() const;

    /**
     * @brief Set Index state code.
     * @param [in] state_code the state code.
     */
    void
    SetStateCode(IndexStateCode state_code);

    /**
     * @brief Failed reason why the index failed to build.
     * @return the failed reason.
     */
    std::string
    FailedReason() const;

    /**
     * @brief Set Failure resaon.
     * @param [in] failed_reason the failed reason.
     */
    void
    SetFailedReason(std::string failed_reason);

 private:
    IndexStateCode state_code_{IndexStateCode::NONE};
    std::string failed_reason_;
};

/**
 * @brief Index progress. Used by GetIndexBuildProgress().
 */
class MILVUS_SDK_API IndexProgress {
 public:
    /**
     * @brief Get number of indexed rows.
     * Note that indexed rows could be larger than total rows, because some segments will be reindexed
     * after compaction.
     * @return the indexed rows.
     */
    int64_t
    IndexedRows() const;

    /**
     * @brief Set number of indexed rows.
     * @param [in] indexed_rows the indexed rows.
     */
    void
    SetIndexedRows(int64_t indexed_rows);

    /**
     * @brief Get number of total rows.
     * @return the total rows.
     */
    int64_t
    TotalRows() const;

    /**
     * @brief Set number of total rows.
     * @param [in] total_rows the total rows.
     */
    void
    SetTotalRows(int64_t total_rows);

 private:
    int64_t indexed_rows_ = 0;
    int64_t total_rows_ = 0;
};

}  // namespace milvus

namespace std {
MILVUS_SDK_API std::string to_string(milvus::IndexStateCode);
}  // namespace std
