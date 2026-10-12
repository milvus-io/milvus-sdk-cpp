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
#include <vector>

#include "./BaseImportRequest.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Zilliz Cloud only. Request for importing data from a storage bucket into a Zilliz cloud instance.
 */
class MILVUS_SDK_API CloudImportRequest : public BaseImportRequest<CloudImportRequest> {
 public:
    /**
     * @brief Constructor
     */
    CloudImportRequest() = default;

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
    CloudImportRequest&
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
    CloudImportRequest&
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
    CloudImportRequest&
    WithRegionId(const std::string& region_id);

    /**
     * @brief Get the object urls to import.
     * @return the object urls.
     */
    const std::vector<std::vector<std::string>>&
    ObjectUrls() const;

    /**
     * @brief Set the object urls to import.
     * @param [in] object_urls the object urls.
     */
    void
    SetObjectUrls(std::vector<std::vector<std::string>>&& object_urls);

    /**
     * @brief Set the object urls to import.
     * @param [in] object_urls the object urls.
     */
    CloudImportRequest&
    WithObjectUrls(std::vector<std::vector<std::string>>&& object_urls);

    /**
     * @brief Get the deprecated single object url.
     * @deprecated Use ObjectUrls() instead.
     * @return the object url.
     */
    [[deprecated("Use ObjectUrls() instead")]] const std::string&
    ObjectUrl() const;

    /**
     * @brief Set the deprecated single object url.
     * @deprecated Use SetObjectUrls() instead.
     * @param [in] object_url the object url.
     */
    [[deprecated("Use SetObjectUrls() instead")]] void
    SetObjectUrl(const std::string& object_url);

    /**
     * @brief Set the deprecated single object url.
     * @deprecated Use WithObjectUrls() instead.
     * @param [in] object_url the object url.
     */
    [[deprecated("Use WithObjectUrls() instead")]] CloudImportRequest&
    WithObjectUrl(const std::string& object_url);

    /**
     * @brief Get the access key for the storage bucket.
     * @return the access key.
     */
    const std::string&
    AccessKey() const;

    /**
     * @brief Set the access key.
     * @param [in] access_key the access key.
     */
    void
    SetAccessKey(const std::string& access_key);

    /**
     * @brief Set the access key.
     * @param [in] access_key the access key.
     */
    CloudImportRequest&
    WithAccessKey(const std::string& access_key);

    /**
     * @brief Get the secret key for the storage bucket.
     * @return the secret key.
     */
    const std::string&
    SecretKey() const;

    /**
     * @brief Set the secret key.
     * @param [in] secret_key the secret key.
     */
    void
    SetSecretKey(const std::string& secret_key);

    /**
     * @brief Set the secret key.
     * @param [in] secret_key the secret key.
     */
    CloudImportRequest&
    WithSecretKey(const std::string& secret_key);

    /**
     * @brief Get the token for short-term credentials.
     * @return the token.
     */
    const std::string&
    Token() const;

    /**
     * @brief Set the token for short-term credentials.
     * @param [in] token the token.
     */
    void
    SetToken(const std::string& token);

    /**
     * @brief Set the token for short-term credentials.
     * @param [in] token the token.
     */
    CloudImportRequest&
    WithToken(const std::string& token);

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
    std::vector<std::vector<std::string>> object_urls_;
    std::string object_url_;
    std::string access_key_;
    std::string secret_key_;
    std::string token_;
};

}  // namespace milvus
