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

#include "milvus/Export.h"
#include "milvus/types/RunningRequestInfo.h"
#include "milvus/types/RunningRequestNodeResult.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::CancelRequests()
 *
 * Every requested id is in exactly one of Canceled(), NotFound() and Undetermined().
 * Cancellation is asynchronous: a canceled request stops at its next cancellation check,
 * usually within milliseconds, and until then ListRunningRequests() shows it in state
 * Canceling.
 */
class MILVUS_SDK_API CancelRequestsResponse {
 public:
    /**
     * @brief Constructor
     */
    CancelRequestsResponse() = default;

    /**
     * @brief The canceled requests, each as it was at the moment it was canceled.
     * @return the canceled requests.
     */
    const RunningRequestInfos&
    Canceled() const;

    /**
     * @brief Set the canceled requests.
     * @param [in] canceled the canceled requests.
     */
    void
    SetCanceled(RunningRequestInfos&& canceled);

    /**
     * @brief The ids no proxy holds, known because every proxy answered: the request
     * already finished, never existed, or was issued by another user.
     * @return the not-found ids.
     */
    const std::vector<std::string>&
    NotFound() const;

    /**
     * @brief Set the not-found ids.
     * @param [in] not_found the not-found ids.
     */
    void
    SetNotFound(const std::vector<std::string>& not_found);

    /**
     * @brief Set the not-found ids.
     * @param [in] not_found the not-found ids.
     */
    void
    SetNotFound(std::vector<std::string>&& not_found);

    /**
     * @brief The ids not canceled while some proxy did not answer: each may be running on
     * that proxy or may have finished. Retry the cancel, or list again, once the proxy is
     * back.
     * @return the undetermined ids.
     */
    const std::vector<std::string>&
    Undetermined() const;

    /**
     * @brief Set the undetermined ids.
     * @param [in] undetermined the undetermined ids.
     */
    void
    SetUndetermined(const std::vector<std::string>& undetermined);

    /**
     * @brief Set the undetermined ids.
     * @param [in] undetermined the undetermined ids.
     */
    void
    SetUndetermined(std::vector<std::string>&& undetermined);

    /**
     * @brief The outcome on each proxy.
     * @return the node results.
     */
    const RunningRequestNodeResults&
    NodeResults() const;

    /**
     * @brief Set the outcome on each proxy.
     * @param [in] node_results the node results.
     */
    void
    SetNodeResults(RunningRequestNodeResults&& node_results);

 private:
    RunningRequestInfos canceled_;
    std::vector<std::string> not_found_;
    std::vector<std::string> undetermined_;
    RunningRequestNodeResults node_results_;
};

}  // namespace milvus
