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
 * @brief Request for describing, committing or aborting a bulk import job on a Milvus server.
 */
class MILVUS_SDK_API MilvusDescribeImportRequest : public BaseDescribeImportRequest<MilvusDescribeImportRequest> {
 public:
    /**
     * @brief Constructor
     */
    MilvusDescribeImportRequest() = default;

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
    MilvusDescribeImportRequest&
    WithJobId(const std::string& job_id);

    /**
     * @brief Serialize the request into the REST import job payload.
     * @return the payload.
     */
    nlohmann::json
    ToJson() const override;

 private:
    std::string job_id_;
};

}  // namespace milvus
