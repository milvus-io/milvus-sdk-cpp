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

#include "milvus/Export.h"
#include "milvus/types/RunningRequestInfo.h"
#include "milvus/types/RunningRequestNodeResult.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::ListRunningRequests()
 *
 * The call succeeds as long as at least one proxy answered; check NodeResults() to tell
 * a complete answer from a partial one.
 */
class MILVUS_SDK_API ListRunningRequestsResponse {
 public:
    /**
     * @brief Constructor
     */
    ListRunningRequestsResponse() = default;

    /**
     * @brief The requests running on the proxies that answered.
     * @return the requests.
     */
    const RunningRequestInfos&
    Requests() const;

    /**
     * @brief Set the requests running on the proxies that answered.
     * @param [in] requests the requests.
     */
    void
    SetRequests(RunningRequestInfos&& requests);

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
    RunningRequestInfos requests_;
    RunningRequestNodeResults node_results_;
};

}  // namespace milvus
