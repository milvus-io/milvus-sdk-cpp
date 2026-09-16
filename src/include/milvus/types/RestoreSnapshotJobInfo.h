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
#include "milvus/types/RestoreSnapshotState.h"

namespace milvus {

/**
 * @brief Information of a snapshot restore job, returned by MilvusClientV2::GetRestoreSnapshotState().
 */
class MILVUS_SDK_API RestoreSnapshotJobInfo {
 public:
    /**
     * @brief Get the name of the snapshot being restored.
     * @return snapshot name.
     */
    const std::string&
    SnapshotName() const;

    /**
     * @brief Set the name of the snapshot being restored.
     * @param [in] snapshot_name snapshot name.
     */
    void
    SetSnapshotName(std::string snapshot_name);

    /**
     * @brief Get the source database name of the restore job.
     * @return source database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set the source database name of the restore job.
     * @param [in] db_name source database name.
     */
    void
    SetDatabaseName(std::string db_name);

    /**
     * @brief Get the source collection name of the restore job.
     * @return source collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set the source collection name of the restore job.
     * @param [in] collection_name source collection name.
     */
    void
    SetCollectionName(std::string collection_name);

    /**
     * @brief Get the restore job identifier.
     * @return job identifier.
     */
    int64_t
    JobID() const;

    /**
     * @brief Set the restore job identifier.
     * @param [in] job_id job identifier.
     */
    void
    SetJobID(int64_t job_id);

    /**
     * @brief Get the current state of the restore job.
     * @return restore job state.
     */
    RestoreSnapshotStateCode
    State() const;

    /**
     * @brief Set the current state of the restore job.
     * @param [in] state restore job state.
     */
    void
    SetState(RestoreSnapshotStateCode state);

    /**
     * @brief Get the restore progress in percent (0 to 100).
     * @return progress in percent.
     */
    int32_t
    Progress() const;

    /**
     * @brief Set the restore progress in percent (0 to 100).
     * @param [in] progress progress in percent.
     */
    void
    SetProgress(int32_t progress);

    /**
     * @brief Get the failure reason of the restore job, empty when the job has not failed.
     * @return failure reason.
     */
    const std::string&
    Reason() const;

    /**
     * @brief Set the failure reason of the restore job.
     * @param [in] reason failure reason.
     */
    void
    SetReason(std::string reason);

    /**
     * @brief Get the start timestamp of the restore job.
     * @return start timestamp.
     */
    uint64_t
    StartTime() const;

    /**
     * @brief Set the start timestamp of the restore job.
     * @param [in] start_time start timestamp.
     */
    void
    SetStartTime(uint64_t start_time);

    /**
     * @brief Get the elapsed time of the restore job in milliseconds.
     * @return elapsed time in milliseconds.
     */
    uint64_t
    TimeCost() const;

    /**
     * @brief Set the elapsed time of the restore job in milliseconds.
     * @param [in] time_cost elapsed time in milliseconds.
     */
    void
    SetTimeCost(uint64_t time_cost);

 private:
    int64_t job_id_{0};
    std::string snapshot_name_;
    std::string db_name_;
    std::string collection_name_;
    RestoreSnapshotStateCode state_{RestoreSnapshotStateCode::UNKNOWN};
    int32_t progress_{0};
    std::string reason_;
    uint64_t start_time_{0};
    uint64_t time_cost_{0};
};

}  // namespace milvus
