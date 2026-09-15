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

#include "../../types/QueryResults.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Query()
 */
class MILVUS_SDK_API QueryResponse {
 public:
    /**
     * @brief Constructor
     */
    QueryResponse() = default;

    /**
     * @brief Get result of query operation.
     * @return the results.
     */
    const QueryResults&
    Results() const;

    /**
     * @brief Set result of query operation.
     * @param [in] results the results.
     */
    void
    SetResults(QueryResults&& results);

    /**
     * @brief Get the session timestamp of the query, used to guarantee consistency in a session.
     * @return the session ts.
     */
    uint64_t
    SessionTs() const;

    /**
     * @brief Set the session timestamp of the query.
     * @param [in] session_ts the session ts.
     */
    void
    SetSessionTs(uint64_t session_ts);

    /**
     * @brief Get the execution cost of the query in cost units, -1 when not reported.
     * @return the cost.
     */
    int64_t
    Cost() const;

    /**
     * @brief Set the execution cost of the query in cost units.
     * @param [in] cost the cost.
     */
    void
    SetCost(int64_t cost);

    /**
     * @brief Get the number of bytes read from remote storage during the query, -1 when not reported.
     * @return the scanned remote bytes.
     */
    int64_t
    ScannedRemoteBytes() const;

    /**
     * @brief Set the number of bytes read from remote storage during the query.
     * @param [in] scanned_remote_bytes the scanned remote bytes.
     */
    void
    SetScannedRemoteBytes(int64_t scanned_remote_bytes);

    /**
     * @brief Get the total number of bytes scanned by the query, -1 when not reported.
     * @return the scanned total bytes.
     */
    int64_t
    ScannedTotalBytes() const;

    /**
     * @brief Set the total number of bytes scanned by the query.
     * @param [in] scanned_total_bytes the scanned total bytes.
     */
    void
    SetScannedTotalBytes(int64_t scanned_total_bytes);

    /**
     * @brief Get the cache hit ratio of the query (0.0 to 1.0), -1.0 when not reported.
     * @return the cache hit ratio.
     */
    float
    CacheHitRatio() const;

    /**
     * @brief Set the cache hit ratio of the query.
     * @param [in] cache_hit_ratio the cache hit ratio.
     */
    void
    SetCacheHitRatio(float cache_hit_ratio);

 private:
    QueryResults results_;
    uint64_t session_ts_{0};
    int64_t cost_{-1};
    int64_t scanned_remote_bytes_{-1};
    int64_t scanned_total_bytes_{-1};
    float cache_hit_ratio_{-1.0f};
};

using GetResponse = QueryResponse;

}  // namespace milvus
