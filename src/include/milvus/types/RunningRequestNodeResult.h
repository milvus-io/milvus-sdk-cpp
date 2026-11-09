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
#include <vector>

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief The outcome of a ListRunningRequests() or CancelRequests() call on one proxy.
 * Both calls are sent to every proxy; a proxy that did not answer contributed nothing to
 * the result, and its requests are missing from it.
 */
class MILVUS_SDK_API RunningRequestNodeResult {
 public:
    /**
     * @brief Constructor
     */
    RunningRequestNodeResult() = default;

    /**
     * @brief The node id of the proxy.
     * @return the node id.
     */
    int64_t
    NodeId() const;

    /**
     * @brief Set the node id of the proxy.
     * @param [in] node_id the node id.
     */
    void
    SetNodeId(int64_t node_id);

    /**
     * @brief Set the node id of the proxy.
     * @param [in] node_id the node id.
     */
    RunningRequestNodeResult&
    WithNodeId(int64_t node_id);

    /**
     * @brief The error code of the proxy's answer; 0 means the proxy answered.
     * @return the error code.
     */
    int64_t
    ErrorCode() const;

    /**
     * @brief Set the error code of the proxy's answer.
     * @param [in] error_code the error code.
     */
    void
    SetErrorCode(int64_t error_code);

    /**
     * @brief Set the error code of the proxy's answer.
     * @param [in] error_code the error code.
     */
    RunningRequestNodeResult&
    WithErrorCode(int64_t error_code);

    /**
     * @brief The error message when the proxy did not answer.
     * @return the reason.
     */
    const std::string&
    Reason() const;

    /**
     * @brief Set the error message when the proxy did not answer.
     * @param [in] reason the reason.
     */
    void
    SetReason(const std::string& reason);

    /**
     * @brief Set the error message when the proxy did not answer.
     * @param [in] reason the reason.
     */
    RunningRequestNodeResult&
    WithReason(const std::string& reason);

    /**
     * @brief Whether the proxy is too old to serve the call, as during a rolling upgrade.
     * Such a proxy holds no listable requests.
     * @return whether the proxy does not implement the call.
     */
    bool
    Unimplemented() const;

    /**
     * @brief Set whether the proxy is too old to serve the call.
     * @param [in] unimplemented whether the proxy does not implement the call.
     */
    void
    SetUnimplemented(bool unimplemented);

    /**
     * @brief Set whether the proxy is too old to serve the call.
     * @param [in] unimplemented whether the proxy does not implement the call.
     */
    RunningRequestNodeResult&
    WithUnimplemented(bool unimplemented);

    /**
     * @brief Whether the proxy answered the call, i.e. the error code is 0.
     * @return whether the proxy answered.
     */
    bool
    IsSuccess() const;

 private:
    int64_t node_id_ = 0;
    int64_t error_code_ = 0;
    std::string reason_;
    bool unimplemented_ = false;
};

/**
 * @brief RunningRequestNodeResult objects array
 */
using RunningRequestNodeResults = std::vector<RunningRequestNodeResult>;

}  // namespace milvus
