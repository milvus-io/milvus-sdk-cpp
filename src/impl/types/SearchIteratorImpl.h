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

#include <list>
#include <memory>
#include <unordered_map>

#include "../MilvusConnection.h"
#include "milvus/request/dql/SearchIteratorRequest.h"
#include "milvus/types/FieldSchema.h"
#include "milvus/types/Iterator.h"
#include "milvus/types/IteratorArguments.h"
#include "milvus/types/MetricType.h"
#include "milvus/types/RetryParam.h"

namespace milvus {

template <typename T>
class SearchIteratorImpl : public SearchIterator {
 public:
    SearchIteratorImpl(const MilvusConnectionPtr& connection, const T& args, const RetryParam& retry_param);

    Status
    Next(SingleResult& results) final;

    Status
    Init();

    static Status
    CheckInput(const FieldDataPtr& vectors, const std::unordered_map<std::string, std::string>& params,
               int64_t batch_size, MetricType metric_type);

    static uint64_t
    CachedCount(const std::list<SingleResultPtr>& cache);

    static Status
    FetchPageFromCache(std::list<SingleResultPtr>& cache, const std::set<std::string>& output_fields, int64_t count,
                       SingleResult& results);

 private:
    static bool
    MetricsPositiveRelated(MetricType metric_type);

    Status
    initSearchIterator();

    void
    updateTailDistance(const SingleResultPtr& results);

    void
    updateWidth(const SingleResult& results);

    Status
    updateFilteredIds(const SingleResultPtr& results);

    int64_t
    extendLimit(bool extend_batch_size);

    Status
    executeSearch(const std::string& filter, bool extend_batch_size, SingleResultPtr& results);

    bool
    reachedLimit() const;

    void
    nextParams(double range_coefficient);

    std::string
    filteredDuplicatedResultFilter();

    Status
    trySearchFill(int64_t count);

 private:
    MilvusConnectionPtr connection_;
    T args_;
    RetryParam retry_param_;
    std::unordered_map<std::string, std::string> original_params_;
    int64_t original_limit_{0};

    uint64_t session_ts_{0};
    uint64_t returned_count_{0};
    double width_{0.0};
    double tail_distance_{0.0};
    std::vector<std::string> filtered_ids_;

    std::list<SingleResultPtr> cache_;
};

// explicitly instantiation of template methods to avoid link error
extern template class SearchIteratorImpl<SearchIteratorArguments>;
extern template class SearchIteratorImpl<SearchIteratorRequest>;

}  // namespace milvus
