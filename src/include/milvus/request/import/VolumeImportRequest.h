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
 * @brief Zilliz Cloud only. Request for importing data from a Zilliz volume into a Zilliz cloud instance.
 */
class MILVUS_SDK_API VolumeImportRequest : public BaseImportRequest<VolumeImportRequest> {
 public:
    /**
     * @brief Constructor
     */
    VolumeImportRequest() = default;

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
    VolumeImportRequest&
    WithClusterId(const std::string& cluster_id);

    /**
     * @brief Get the volume name.
     * @return the volume name.
     */
    const std::string&
    VolumeName() const;

    /**
     * @brief Set the volume name.
     * @param [in] volume_name the volume name.
     */
    void
    SetVolumeName(const std::string& volume_name);

    /**
     * @brief Set the volume name.
     * @param [in] volume_name the volume name.
     */
    VolumeImportRequest&
    WithVolumeName(const std::string& volume_name);

    /**
     * @brief Get the data paths to import.
     * @return the data paths.
     */
    const std::vector<std::vector<std::string>>&
    DataPaths() const;

    /**
     * @brief Set the data paths to import.
     * @param [in] data_paths the data paths.
     */
    void
    SetDataPaths(std::vector<std::vector<std::string>>&& data_paths);

    /**
     * @brief Set the data paths to import.
     * @param [in] data_paths the data paths.
     */
    VolumeImportRequest&
    WithDataPaths(std::vector<std::vector<std::string>>&& data_paths);

    /**
     * @brief Serialize the request into the REST import job payload.
     * @return the payload.
     */
    nlohmann::json
    ToJson() const override;

 private:
    std::string cluster_id_;
    std::string volume_name_;
    std::vector<std::vector<std::string>> data_paths_;
};

}  // namespace milvus
