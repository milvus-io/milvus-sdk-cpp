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

namespace milvus {

/**
 * @brief Used by MilvusClientV2::CheckHealth()
 */
class CheckHealthResponse {
 public:
    /**
     * @brief Constructor
     */
    CheckHealthResponse() = default;

    /**
     * @brief Get whether the milvus server is healthy or not.
     */
    bool
    IsHealthy() const;

    /**
     * @brief Set whether the milvus server is healthy or not.
     */
    void
    SetIsHealthy(bool healthy);

    /**
     * @brief Get the reasons why the milvus server is unhealthy.
     */
    const std::vector<std::string>&
    Reasons() const;

    /**
     * @brief Set the reasons why the milvus server is unhealthy.
     */
    void
    SetReasons(std::vector<std::string>&& reasons);

    /**
     * @brief Get the quota states why the milvus server is unable service.
     */
    const std::vector<std::string>&
    QuotaStates() const;

    /**
     * @brief Set the quota states why the milvus server is unable service.
     */
    void
    SetQuotaStates(std::vector<std::string>&& states);

 private:
    bool is_healthy_{false};
    std::vector<std::string> reasons_;
    std::vector<std::string> quota_states_;
};

}  // namespace milvus
