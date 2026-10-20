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
 * @brief Used by MilvusClientV2::GetRLSPrincipalTags()
 */
class MILVUS_SDK_API GetRLSPrincipalTagsResponse {
 public:
    /**
     * @brief Constructor
     */
    GetRLSPrincipalTagsResponse() = default;

    /**
     * @brief Get the decoded principal tags. Values preserve integer, float and
     * array types.
     * @return the tags.
     */
    const nlohmann::json&
    Tags() const;

    /**
     * @brief Set the decoded principal tags.
     * @param [in] tags the tags.
     */
    void
    SetTags(nlohmann::json&& tags);

    /**
     * @brief Get the application-level principal name.
     * @return the principal name.
     */
    const std::string&
    PrincipalName() const;

    /**
     * @brief Set the application-level principal name.
     * @param [in] principal_name the principal name.
     */
    void
    SetPrincipalName(const std::string& principal_name);

 private:
    nlohmann::json tags_ = nlohmann::json::object();
    std::string principal_name_;
};

}  // namespace milvus
