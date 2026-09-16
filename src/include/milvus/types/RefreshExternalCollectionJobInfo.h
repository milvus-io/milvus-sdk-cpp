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
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <string>

#include "milvus/Export.h"
#include "milvus/types/RefreshExternalCollectionState.h"

namespace milvus {

/**
 * @brief Information of an external collection refresh job.
 */
class MILVUS_SDK_API RefreshExternalCollectionJobInfo {
 public:
    /**
     * @brief Get the refresh job identifier.
     * @return job identifier.
     */
    int64_t
    JobID() const;

    /**
     * @brief Set the refresh job identifier.
     * @param [in] job_id job identifier.
     */
    void
    SetJobID(int64_t job_id);

    /**
     * @brief Get the collection being refreshed.
     * @return collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set the collection being refreshed.
     * @param [in] collection_name collection name.
     */
    void
    SetCollectionName(std::string collection_name);

    /**
     * @brief Get the current state of the refresh job.
     * @return refresh job state.
     */
    RefreshExternalCollectionStateCode
    State() const;

    /**
     * @brief Set the current state of the refresh job.
     * @param [in] state refresh job state.
     */
    void
    SetState(RefreshExternalCollectionStateCode state);

    /**
     * @brief Get the refresh progress in percent (0 to 100).
     * @return progress in percent.
     */
    int32_t
    Progress() const;

    /**
     * @brief Set the refresh progress in percent (0 to 100).
     * @param [in] progress progress in percent.
     */
    void
    SetProgress(int32_t progress);

    /**
     * @brief Get the failure reason of the refresh job, empty when the job has not failed.
     * @return failure reason.
     */
    const std::string&
    Reason() const;

    /**
     * @brief Set the failure reason of the refresh job.
     * @param [in] reason failure reason.
     */
    void
    SetReason(std::string reason);

    /**
     * @brief Get the external data source, e.g. an S3 path.
     * @return external source.
     */
    const std::string&
    ExternalSource() const;

    /**
     * @brief Set the external data source.
     * @param [in] external_source external source.
     */
    void
    SetExternalSource(std::string external_source);

    /**
     * @brief Get the start timestamp of the refresh job.
     * @return start timestamp.
     */
    uint64_t
    StartTime() const;

    /**
     * @brief Set the start timestamp of the refresh job.
     * @param [in] start_time start timestamp.
     */
    void
    SetStartTime(uint64_t start_time);

    /**
     * @brief Get the end timestamp of the refresh job.
     * @return end timestamp.
     */
    uint64_t
    EndTime() const;

    /**
     * @brief Set the end timestamp of the refresh job.
     * @param [in] end_time end timestamp.
     */
    void
    SetEndTime(uint64_t end_time);

    /**
     * @brief Get the external file specification.
     * @return external specification.
     */
    const nlohmann::json&
    ExternalSpec() const;

    /**
     * @brief Set the external file specification.
     * @param [in] external_spec external specification.
     */
    void
    SetExternalSpec(const nlohmann::json& external_spec);

 private:
    int64_t job_id_{0};
    std::string collection_name_;
    RefreshExternalCollectionStateCode state_{RefreshExternalCollectionStateCode::PENDING};
    int32_t progress_{0};
    std::string reason_;
    std::string external_source_;
    uint64_t start_time_{0};
    uint64_t end_time_{0};
    nlohmann::json external_spec_;
};

}  // namespace milvus
