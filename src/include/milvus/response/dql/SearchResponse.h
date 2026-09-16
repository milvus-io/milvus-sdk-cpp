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

#include "../../types/AggregationBucket.h"
#include "../../types/SearchResults.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Search()
 */
class MILVUS_SDK_API SearchResponse {
 public:
    /**
     * @brief Constructor
     */
    SearchResponse() = default;

    /**
     * @brief Get result of search operation.
     * @return the results.
     */
    const SearchResults&
    Results() const;

    /**
     * @brief Set result of search operation.
     * @param [in] results the results.
     */
    void
    SetResults(SearchResults&& results);

    /**
     * @brief Get the session timestamp of the search, used to guarantee consistency in a session.
     * @return the session ts.
     */
    uint64_t
    SessionTs() const;

    /**
     * @brief Set the session timestamp of the search.
     * @param [in] session_ts the session ts.
     */
    void
    SetSessionTs(uint64_t session_ts);

    /**
     * @brief Get the execution cost of the search in cost units, -1 when not reported.
     * @return the cost.
     */
    int64_t
    Cost() const;

    /**
     * @brief Set the execution cost of the search in cost units.
     * @param [in] cost the cost.
     */
    void
    SetCost(int64_t cost);

    /**
     * @brief Get the number of bytes read from remote storage during the search, -1 when not reported.
     * @return the scanned remote bytes.
     */
    int64_t
    ScannedRemoteBytes() const;

    /**
     * @brief Set the number of bytes read from remote storage during the search.
     * @param [in] scanned_remote_bytes the scanned remote bytes.
     */
    void
    SetScannedRemoteBytes(int64_t scanned_remote_bytes);

    /**
     * @brief Get the total number of bytes scanned by the search, -1 when not reported.
     * @return the scanned total bytes.
     */
    int64_t
    ScannedTotalBytes() const;

    /**
     * @brief Set the total number of bytes scanned by the search.
     * @param [in] scanned_total_bytes the scanned total bytes.
     */
    void
    SetScannedTotalBytes(int64_t scanned_total_bytes);

    /**
     * @brief Get the cache hit ratio of the search (0.0 to 1.0), -1.0 when not reported.
     * @return the cache hit ratio.
     */
    float
    CacheHitRatio() const;

    /**
     * @brief Set the cache hit ratio of the search.
     * @param [in] cache_hit_ratio the cache hit ratio.
     */
    void
    SetCacheHitRatio(float cache_hit_ratio);

    /**
     * @brief Get aggregation buckets grouped by search query.
     *
     * The outer vector follows query order, and each inner vector contains that query's buckets. Queries with no
     * buckets retain an empty inner vector so their indexes remain aligned with the search request.
     * @return the aggregation buckets.
     */
    const milvus::AggregationBuckets&
    AggregationBuckets() const;

    /**
     * @brief Set aggregation buckets grouped by search query.
     * @param [in] aggregation_buckets the aggregation buckets.
     */
    void
    SetAggregationBuckets(milvus::AggregationBuckets&& aggregation_buckets);

 private:
    SearchResults results_;
    uint64_t session_ts_{0};
    int64_t cost_{-1};
    int64_t scanned_remote_bytes_{-1};
    int64_t scanned_total_bytes_{-1};
    float cache_hit_ratio_{-1.0f};
    milvus::AggregationBuckets aggregation_buckets_;
};

using HybridSearchResponse = SearchResponse;  // hybrid search and search have the same result

}  // namespace milvus
