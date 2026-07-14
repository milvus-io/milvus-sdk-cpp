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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "milvus/Export.h"
#include "milvus/Status.h"

namespace milvus {

enum class AggregationDirection { ASC, DESC };

enum class AggregationMetricOp { AVG, SUM, COUNT, MIN, MAX };

struct MILVUS_SDK_API AggregationMetric {
    AggregationMetric() = default;
    AggregationMetric(AggregationMetricOp op, std::string field_name);

    AggregationMetricOp op{AggregationMetricOp::COUNT};
    std::string field_name;
};

struct MILVUS_SDK_API AggregationOrder {
    AggregationOrder() = default;
    AggregationOrder(std::string key, AggregationDirection direction, bool null_first = false);

    std::string key;
    AggregationDirection direction{AggregationDirection::ASC};
    bool null_first{false};
};

struct MILVUS_SDK_API AggregationSort {
    AggregationSort() = default;
    AggregationSort(std::string field_name, AggregationDirection direction, bool null_first = false);

    std::string field_name;
    AggregationDirection direction{AggregationDirection::ASC};
    bool null_first{false};
};

class MILVUS_SDK_API AggregationTopHits {
 public:
    AggregationTopHits() = default;
    explicit AggregationTopHits(int64_t size);

    int64_t
    Size() const;

    void
    SetSize(int64_t size);

    AggregationTopHits&
    WithSize(int64_t size);

    const std::vector<AggregationSort>&
    Sorts() const;

    void
    SetSorts(std::vector<AggregationSort>&& sorts);

    AggregationTopHits&
    WithSorts(std::vector<AggregationSort>&& sorts);

    AggregationTopHits&
    AddSort(AggregationSort sort);

    Status
    Validate() const;

 private:
    int64_t size_{0};
    std::vector<AggregationSort> sorts_;
};

using AggregationTopHitsPtr = std::shared_ptr<AggregationTopHits>;

class SearchAggregation;
using SearchAggregationPtr = std::shared_ptr<SearchAggregation>;

class MILVUS_SDK_API SearchAggregation {
 public:
    SearchAggregation() = default;
    SearchAggregation(std::vector<std::string> fields, int64_t size);

    const std::vector<std::string>&
    Fields() const;

    void
    SetFields(std::vector<std::string>&& fields);

    SearchAggregation&
    WithFields(std::vector<std::string>&& fields);

    SearchAggregation&
    AddField(std::string field);

    int64_t
    Size() const;

    void
    SetSize(int64_t size);

    SearchAggregation&
    WithSize(int64_t size);

    const std::map<std::string, AggregationMetric>&
    Metrics() const;

    void
    SetMetrics(std::map<std::string, AggregationMetric>&& metrics);

    SearchAggregation&
    WithMetrics(std::map<std::string, AggregationMetric>&& metrics);

    SearchAggregation&
    AddMetric(std::string alias, AggregationMetric metric);

    const std::vector<AggregationOrder>&
    Orders() const;

    void
    SetOrders(std::vector<AggregationOrder>&& orders);

    SearchAggregation&
    WithOrders(std::vector<AggregationOrder>&& orders);

    SearchAggregation&
    AddOrder(AggregationOrder order);

    const AggregationTopHitsPtr&
    TopHits() const;

    void
    SetTopHits(const AggregationTopHitsPtr& top_hits);

    SearchAggregation&
    WithTopHits(const AggregationTopHitsPtr& top_hits);

    const SearchAggregationPtr&
    SubAggregation() const;

    void
    SetSubAggregation(const SearchAggregationPtr& sub_aggregation);

    SearchAggregation&
    WithSubAggregation(const SearchAggregationPtr& sub_aggregation);

    Status
    Validate() const;

 private:
    std::vector<std::string> fields_;
    int64_t size_{0};
    std::map<std::string, AggregationMetric> metrics_;
    std::vector<AggregationOrder> orders_;
    AggregationTopHitsPtr top_hits_;
    SearchAggregationPtr sub_aggregation_;
};

}  // namespace milvus
