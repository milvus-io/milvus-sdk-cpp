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
#include <string>

namespace milvus {
// const values for internal common usage
constexpr int64_t MAX_BATCH_SIZE = 16384;
constexpr uint64_t ITERATION_MAX_FILTERED_IDS_COUNT = 100000;
constexpr uint64_t ITERATION_MAX_RETRY_TIME = 20;

// const names for internal common usage
extern const char* ROW_COUNT;
extern const char* TOPK;
extern const char* LIMIT;
extern const char* OFFSET;
extern const char* ANNS_FIELD;
extern const char* RADIUS;
extern const char* RANGE_FILTER;
extern const char* IGNORE_GROWING;
extern const char* ROUND_DECIMAL;
extern const char* GROUPBY_FIELD;
extern const char* GROUPBY_SIZE;
extern const char* GROUPBY_STRICT_SIZE;
extern const char* PARAMS;
extern const char* STRATEGY;
extern const char* SCORE;
extern const char* ENABLE_ANALYZER;
extern const char* ENABLE_MATCH;
extern const char* ANALYZER_PARAMS;

extern const char* ITERATOR;
extern const char* REDUCE_STOP_FOR_BEST;
extern const char* COLLECTION_ID;

std::string
GetBuildVersion();

}  // namespace milvus
