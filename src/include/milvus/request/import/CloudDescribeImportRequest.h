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

#include "./BaseDescribeImportRequest.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Zilliz Cloud only. Request for describing, committing or aborting an import job on a Zilliz cloud instance.
 */
class MILVUS_SDK_API CloudDescribeImportRequest : public BaseDescribeImportRequest<CloudDescribeImportRequest> {
 public:
    /**
     * @brief Constructor
     */
    CloudDescribeImportRequest() = default;

    /**
     * @brief Get the cluster id.
     * @return the cluster id.
     */
    const std::string&
    ClusterId() const;

    /**
     * @brief Set the cluster id.
     * @param [in] cluster_id the cluster id.
     */
    void
    SetClusterId(const std::string& cluster_id);

    /**
     * @brief Set the cluster id.
     * @param [in] cluster_id the cluster id.
     */
    CloudDescribeImportRequest&
    WithClusterId(const std::string& cluster_id);

    /**
     * @brief Get the project id, used for project database deployments.
     * @return the project id.
     */
    const std::string&
    ProjectId() const;

    /**
     * @brief Set the project id.
     * @param [in] project_id the project id.
     */
    void
    SetProjectId(const std::string& project_id);

    /**
     * @brief Set the project id.
     * @param [in] project_id the project id.
     */
    CloudDescribeImportRequest&
    WithProjectId(const std::string& project_id);

    /**
     * @brief Get the region id, used for project database deployments.
     * @return the region id.
     */
    const std::string&
    RegionId() const;

    /**
     * @brief Set the region id.
     * @param [in] region_id the region id.
     */
    void
    SetRegionId(const std::string& region_id);

    /**
     * @brief Set the region id.
     * @param [in] region_id the region id.
     */
    CloudDescribeImportRequest&
    WithRegionId(const std::string& region_id);

    /**
     * @brief Get the id of the import job.
     * @return the job id.
     */
    const std::string&
    JobId() const;

    /**
     * @brief Set the id of the import job.
     * @param [in] job_id the job id.
     */
    void
    SetJobId(const std::string& job_id);

    /**
     * @brief Set the id of the import job.
     * @param [in] job_id the job id.
     */
    CloudDescribeImportRequest&
    WithJobId(const std::string& job_id);

    /**
     * @brief Serialize the request into the REST import job payload.
     * @return the payload.
     */
    nlohmann::json
    ToJson() const override;

 private:
    std::string cluster_id_;
    std::string project_id_;
    std::string region_id_;
    std::string job_id_;
};

}  // namespace milvus
