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

#include "milvus/response/dql/SearchResponse.h"

#include <memory>

namespace milvus {

const SearchResults&
SearchResponse::Results() const {
    return results_;
}

void
SearchResponse::SetResults(SearchResults&& results) {
    results_ = std::move(results);
}

uint64_t
SearchResponse::SessionTs() const {
    return session_ts_;
}

void
SearchResponse::SetSessionTs(uint64_t session_ts) {
    session_ts_ = session_ts;
}

int64_t
SearchResponse::Cost() const {
    return cost_;
}

void
SearchResponse::SetCost(int64_t cost) {
    cost_ = cost;
}

int64_t
SearchResponse::ScannedRemoteBytes() const {
    return scanned_remote_bytes_;
}

void
SearchResponse::SetScannedRemoteBytes(int64_t scanned_remote_bytes) {
    scanned_remote_bytes_ = scanned_remote_bytes;
}

int64_t
SearchResponse::ScannedTotalBytes() const {
    return scanned_total_bytes_;
}

void
SearchResponse::SetScannedTotalBytes(int64_t scanned_total_bytes) {
    scanned_total_bytes_ = scanned_total_bytes;
}

float
SearchResponse::CacheHitRatio() const {
    return cache_hit_ratio_;
}

void
SearchResponse::SetCacheHitRatio(float cache_hit_ratio) {
    cache_hit_ratio_ = cache_hit_ratio;
}

const milvus::AggregationBuckets&
SearchResponse::AggregationBuckets() const {
    return aggregation_buckets_;
}

void
SearchResponse::SetAggregationBuckets(milvus::AggregationBuckets&& aggregation_buckets) {
    aggregation_buckets_ = std::move(aggregation_buckets);
}

}  // namespace milvus
