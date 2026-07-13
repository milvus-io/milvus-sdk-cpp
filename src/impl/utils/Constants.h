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
#include <set>
#include <string>

#include "milvus/types/Constants.h"

namespace milvus {
// const values for internal common usage
constexpr int64_t MAX_BATCH_SIZE = 16384;
constexpr uint64_t ITERATION_MAX_FILTERED_IDS_COUNT = 100000;
constexpr uint64_t ITERATION_MAX_RETRY_TIME = 20;
constexpr uint64_t DEFAULT_OPTIMIZE_RPC_TIMEOUT_MS = 60000;

// const names for internal common usage
extern MILVUS_SDK_API const char* ROW_COUNT;
extern MILVUS_SDK_API const char* TOPK;
extern MILVUS_SDK_API const char* LIMIT;
extern MILVUS_SDK_API const char* OFFSET;
extern MILVUS_SDK_API const char* ANNS_FIELD;
extern MILVUS_SDK_API const char* RADIUS;
extern MILVUS_SDK_API const char* RANGE_FILTER;
extern MILVUS_SDK_API const char* IGNORE_GROWING;
extern MILVUS_SDK_API const char* ROUND_DECIMAL;
extern MILVUS_SDK_API const char* GROUPBY_FIELD;
extern MILVUS_SDK_API const char* GROUPBY_SIZE;
extern MILVUS_SDK_API const char* GROUPBY_STRICT_SIZE;
extern MILVUS_SDK_API const char* PARAMS;
extern MILVUS_SDK_API const char* STRATEGY;
extern MILVUS_SDK_API const char* SCORE;
extern MILVUS_SDK_API const char* ELEMENT_OFFSET;
extern MILVUS_SDK_API const char* ENABLE_MATCH;
extern MILVUS_SDK_API const char* ENABLE_ANALYZER;
extern MILVUS_SDK_API const char* ANALYZER_PARAMS;
extern MILVUS_SDK_API const char* MULTI_ANALYZER_PARAMS;
extern MILVUS_SDK_API const char* COLLECTION_ID;
extern MILVUS_SDK_API const char* CLUSTER_ID;
extern MILVUS_SDK_API const char* GUARANTEE_TIMESTAMP;
extern MILVUS_SDK_API const char* REDUCE_STOP_FOR_BEST;
extern MILVUS_SDK_API const char* ITERATOR_FIELD;
extern MILVUS_SDK_API const char* ITERATOR_SESSION_TS_FIELD;
extern MILVUS_SDK_API const char* ITER_SEARCH_V2_KEY;
extern MILVUS_SDK_API const char* ITER_SEARCH_BATCH_SIZE_KEY;
extern MILVUS_SDK_API const char* ITER_SEARCH_LAST_BOUND_KEY;
extern MILVUS_SDK_API const char* ITER_SEARCH_ID_KEY;
extern MILVUS_SDK_API const char* RERANKER;
extern MILVUS_SDK_API const char* RANDOM_SCORE;

std::string
GetBuildVersion();

}  // namespace milvus
