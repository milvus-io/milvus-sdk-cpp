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

/**
 * @brief The key value of an aggregation bucket, defined by the group-by fields of one query.
 */
struct MILVUS_SDK_API AggregationBucketKey {
    /** @brief field id of the group-by field. */
    int64_t field_id{0};
    /** @brief field name of the group-by field. */
    std::string field_name;
    /** @brief group-by field value of the bucket. */
    nlohmann::json value;
};

/**
 * @brief One top hit document of an aggregation bucket.
 */
struct MILVUS_SDK_API AggregationHit {
    /** @brief primary key of the document. */
    nlohmann::json id;
    /** @brief relevance score of the document. */
    float score{0.0f};
    /** @brief field values of the document, keyed by field name. */
    std::unordered_map<std::string, nlohmann::json> fields;
    /** @brief field ids of the document, keyed by field name. */
    std::unordered_map<std::string, int64_t> field_ids;
};

/**
 * @brief One aggregation bucket, keyed by the group-by field values.
 */
struct MILVUS_SDK_API AggregationBucket {
    /** @brief bucket key values, aligned with the group-by fields. */
    std::vector<AggregationBucketKey> key;
    /** @brief number of documents in the bucket. */
    int64_t count{0};
    /** @brief metric values, keyed by metric alias. */
    std::unordered_map<std::string, nlohmann::json> metrics;
    /** @brief top hits of the bucket. */
    std::vector<AggregationHit> hits;
    /** @brief nested sub-buckets. */
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
