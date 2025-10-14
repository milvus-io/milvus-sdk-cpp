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

#include "Constants.h"

#include "../version.h"
#include "milvus/types/Constants.h"

namespace milvus {
// const names for internal common usage
const char* ROW_COUNT = "row_count";
const char* TOPK = "topk";
const char* LIMIT = "limit";
const char* OFFSET = "offset";
const char* ANNS_FIELD = "anns_field";
const char* RADIUS = "radius";
const char* RANGE_FILTER = "range_filter";
const char* IGNORE_GROWING = "ignore_growing";
const char* ROUND_DECIMAL = "round_decimal";
const char* GROUPBY_FIELD = "group_by_field";
const char* PARAMS = "params";
const char* STRATEGY = "strategy";
const char* SCORE = "score";
const char* ENABLE_ANALYZER = "enable_analyzer";
const char* ENABLE_MATCH = "enable_match";
const char* ANALYZER_PARAMS = "analyzer_params";
const char* ITERATOR = "iterator";
const char* REDUCE_STOP_FOR_BEST = "reduce_stop_for_best";
const char* COLLECTION_ID = "collection_id";

// const names for external common usage
const char* INDEX_TYPE = "index_type";
const char* METRIC_TYPE = "metric_type";
const char* NLIST = "nlist";
const char* NPROBE = "nprobe";
const char* DIM = "dim";
const char* MAX_LENGTH = "max_length";
const char* MAX_CAPACITY = "max_capacity";
const char* DYNAMIC_FIELD = "$meta";
const char* SPARSE_INDICES = "indices";
const char* SPARSE_VALUES = "values";
const char* MMAP_ENABLED = "mmap.enabled";
const char* COLLECTION_TTL_SECONDS = "collection.ttl.seconds";
const char* COLLECTION_REPLICA_NUMBER = "collection.replica.number";
const char* COLLECTION_RESOURCE_GROUPS = "collection.resource_groups";
const char* DATABASE_REPLICA_NUMBER = "database.replica.number";
const char* DATABASE_RESOURCE_GROUPS = "database.resource_groups";

std::string
GetBuildVersion() {
    return std::string(MILVUS_SDK_VERSION) + "-" + std::string(CMAKE_BUILD_TYPE);
}

}  // namespace milvus
