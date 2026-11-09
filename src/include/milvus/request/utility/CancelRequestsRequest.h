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

namespace milvus {

/**
 * @brief Used by MilvusClientV2::CancelRequests()
 */
class MILVUS_SDK_API CancelRequestsRequest {
 public:
    /**
     * @brief Constructor
     */
    CancelRequestsRequest() = default;

    /**
     * @brief The ids of the requests to cancel, exactly as returned by
     * ListRunningRequests(); must not be empty.
     * @return the request ids.
     */
    const std::vector<std::string>&
    RequestIds() const;

    /**
     * @brief Set the ids of the requests to cancel.
     * @param [in] request_ids the request ids.
     */
    void
    SetRequestIds(const std::vector<std::string>& request_ids);

    /**
     * @brief Set the ids of the requests to cancel.
     * @param [in] request_ids the request ids.
     */
    void
    SetRequestIds(std::vector<std::string>&& request_ids);

    /**
     * @brief Set the ids of the requests to cancel.
     * @param [in] request_ids the request ids.
     */
    CancelRequestsRequest&
    WithRequestIds(std::vector<std::string>&& request_ids);

    /**
     * @brief Add a request id to cancel.
     * @param [in] request_id the request id.
     */
    CancelRequestsRequest&
    AddRequestId(const std::string& request_id);

    /**
     * @brief A free-text reason, recorded in the server's audit log and returned to the
     * client whose request is canceled.
     * @return the reason.
     */
    const std::string&
    Reason() const;

    /**
     * @brief Set the cancellation reason.
     * @param [in] reason the reason.
     */
    void
    SetReason(const std::string& reason);

    /**
     * @brief Set the cancellation reason.
     * @param [in] reason the reason.
     */
    CancelRequestsRequest&
    WithReason(const std::string& reason);

    /**
     * @brief When set, only requests issued by this user are canceled; the ids of other
     * users' requests are reported as not found. A caller without the CancelRequests
     * privilege may only cancel their own requests: empty then means the caller, and any
     * other name is denied.
     * @return the user.
     */
    const std::string&
    User() const;

    /**
     * @brief Set the user whose requests are canceled.
     * @param [in] user the user.
     */
    void
    SetUser(const std::string& user);

    /**
     * @brief Set the user whose requests are canceled.
     * @param [in] user the user.
     */
    CancelRequestsRequest&
    WithUser(const std::string& user);

 private:
    std::vector<std::string> request_ids_;
    std::string reason_;
    std::string user_;
};

}  // namespace milvus
