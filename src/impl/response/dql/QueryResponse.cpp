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

#include "milvus/response/dql/QueryResponse.h"

#include <memory>

namespace milvus {

const QueryResults&
QueryResponse::Results() const {
    return results_;
}

void
QueryResponse::SetResults(QueryResults&& results) {
    results_ = std::move(results);
}

uint64_t
QueryResponse::SessionTs() const {
    return session_ts_;
}

void
QueryResponse::SetSessionTs(uint64_t session_ts) {
    session_ts_ = session_ts;
}

int64_t
QueryResponse::Cost() const {
    return cost_;
}

void
QueryResponse::SetCost(int64_t cost) {
    cost_ = cost;
}

int64_t
QueryResponse::ScannedRemoteBytes() const {
    return scanned_remote_bytes_;
}

void
QueryResponse::SetScannedRemoteBytes(int64_t scanned_remote_bytes) {
    scanned_remote_bytes_ = scanned_remote_bytes;
}

int64_t
QueryResponse::ScannedTotalBytes() const {
    return scanned_total_bytes_;
}

void
QueryResponse::SetScannedTotalBytes(int64_t scanned_total_bytes) {
    scanned_total_bytes_ = scanned_total_bytes;
}

float
QueryResponse::CacheHitRatio() const {
    return cache_hit_ratio_;
}

void
QueryResponse::SetCacheHitRatio(float cache_hit_ratio) {
    cache_hit_ratio_ = cache_hit_ratio;
}

}  // namespace milvus
