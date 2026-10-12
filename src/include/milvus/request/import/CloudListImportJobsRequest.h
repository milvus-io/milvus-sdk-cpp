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

#include "./BaseListImportJobsRequest.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Zilliz Cloud only. Request for listing import jobs on a Zilliz cloud instance.
 */
class MILVUS_SDK_API CloudListImportJobsRequest : public BaseListImportJobsRequest<CloudListImportJobsRequest> {
 public:
    /**
     * @brief Constructor
     */
    CloudListImportJobsRequest() = default;

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
    CloudListImportJobsRequest&
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
    CloudListImportJobsRequest&
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
    CloudListImportJobsRequest&
    WithRegionId(const std::string& region_id);

    /**
     * @brief Get the page size.
     * @return the page size.
     */
    int64_t
    PageSize() const;

    /**
     * @brief Set the page size.
     * @param [in] page_size the page size.
     */
    void
    SetPageSize(int64_t page_size);

    /**
     * @brief Set the page size.
     * @param [in] page_size the page size.
     */
    CloudListImportJobsRequest&
    WithPageSize(int64_t page_size);

    /**
     * @brief Get the current page number.
     * @return the current page.
     */
    int64_t
    CurrentPage() const;

    /**
     * @brief Set the current page number.
     * @param [in] current_page the current page.
     */
    void
    SetCurrentPage(int64_t current_page);

    /**
     * @brief Set the current page number.
     * @param [in] current_page the current page.
     */
    CloudListImportJobsRequest&
    WithCurrentPage(int64_t current_page);

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
    int64_t page_size_{0};
    int64_t current_page_{0};
};

}  // namespace milvus
