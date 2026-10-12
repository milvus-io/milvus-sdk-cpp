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
#include "milvus/thirdparty/nlohmann/json.hpp"

namespace milvus {

/**
 * @brief Response of a bulk import REST API call.
 */
class MILVUS_SDK_API BulkImportResponse {
 public:
    /**
     * @brief Constructor
     */
    BulkImportResponse() = default;

    /**
     * @brief Get the raw JSON body of the REST response.
     * @return the raw JSON.
     */
    const nlohmann::json&
    RawJson() const;

    /**
     * @brief Set the raw JSON body of the REST response.
     * @param [in] json the raw JSON.
     */
    void
    SetRawJson(nlohmann::json&& json);

    /**
     * @brief Get the REST envelope code.
     * @return the code.
     */
    int64_t
    Code() const;

    /**
     * @brief Get the REST envelope message.
     * @return the message.
     */
    std::string
    Message() const;

    /**
     * @brief Get the REST envelope data object.
     * @return the data.
     */
    const nlohmann::json&
    Data() const;

    /**
     * @brief Get the id of the created import job, empty for other operations.
     * @return the job id.
     */
    std::string
    JobId() const;

 private:
    nlohmann::json json_;
};

}  // namespace milvus
