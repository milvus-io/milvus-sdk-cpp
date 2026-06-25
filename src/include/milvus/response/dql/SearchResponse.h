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
     */
    const SearchResults&
    Results() const;

    /**
     * @brief Set result of search operation.
     */
    void
    SetResults(SearchResults&& results);

    uint64_t
    SessionTs() const;

    void
    SetSessionTs(uint64_t session_ts);

    int64_t
    Cost() const;

    void
    SetCost(int64_t cost);

    int64_t
    ScannedRemoteBytes() const;

    void
    SetScannedRemoteBytes(int64_t scanned_remote_bytes);

    int64_t
    ScannedTotalBytes() const;

    void
    SetScannedTotalBytes(int64_t scanned_total_bytes);

    float
    CacheHitRatio() const;

    void
    SetCacheHitRatio(float cache_hit_ratio);

 private:
    SearchResults results_;
    uint64_t session_ts_{0};
    int64_t cost_{-1};
    int64_t scanned_remote_bytes_{-1};
    int64_t scanned_total_bytes_{-1};
    float cache_hit_ratio_{-1.0f};
};

using HybridSearchResponse = SearchResponse;  // hybrid search and search have the same result

}  // namespace milvus
