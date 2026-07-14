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
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "milvus/Export.h"

namespace milvus {

struct MILVUS_SDK_API AggregationBucketKey {
    int64_t field_id{0};
    std::string field_name;
    nlohmann::json value;
};

struct MILVUS_SDK_API AggregationHit {
    nlohmann::json id;
    float score{0.0f};
    std::unordered_map<std::string, nlohmann::json> fields;
    std::unordered_map<std::string, int64_t> field_ids;
};

struct MILVUS_SDK_API AggregationBucket {
    std::vector<AggregationBucketKey> key;
    int64_t count{0};
    std::unordered_map<std::string, nlohmann::json> metrics;
    std::vector<AggregationHit> hits;
    std::vector<AggregationBucket> sub_groups;
};

/**
 * @brief Aggregation buckets grouped by search query.
 *
 * The outer vector follows the request's query order. Each inner vector contains the buckets returned for that
 * query. An aggregation response includes an empty inner vector for every query that has no buckets, preserving
 * alignment between query indexes and aggregation results.
 */
using AggregationBuckets = std::vector<std::vector<AggregationBucket>>;

}  // namespace milvus
