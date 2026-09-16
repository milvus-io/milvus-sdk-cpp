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

/**
 * @brief Sort direction of aggregation buckets.
 */
enum class AggregationDirection { ASC, DESC };

/**
 * @brief Metric operator of an aggregation metric.
 */
enum class AggregationMetricOp { AVG, SUM, COUNT, MIN, MAX };

/**
 * @brief An aggregation metric: an operator applied to a field, stored under an alias.
 */
struct MILVUS_SDK_API AggregationMetric {
    /**
     * @brief Constructor
     */
    AggregationMetric() = default;

    /**
     * @brief Construct an aggregation metric.
     *
     * @param [in] op metric operator.
     * @param [in] field_name target field; use "*" for COUNT over all fields.
     */
    AggregationMetric(AggregationMetricOp op, std::string field_name);

    /** @brief metric operator. */
    AggregationMetricOp op{AggregationMetricOp::COUNT};
    /** @brief target field name. */
    std::string field_name;
};

/**
 * @brief An ordering of aggregation buckets by a metric alias or the bucket key.
 */
struct MILVUS_SDK_API AggregationOrder {
    /**
     * @brief Constructor
     */
    AggregationOrder() = default;

    /**
     * @brief Construct an aggregation order.
     *
     * @param [in] key metric alias, "_count", or "_key".
     * @param [in] direction sort direction.
     * @param [in] null_first whether buckets with a null value sort first.
     */
    AggregationOrder(std::string key, AggregationDirection direction, bool null_first = false);

    /** @brief metric alias, "_count", or "_key". */
    std::string key;
    /** @brief sort direction. */
    AggregationDirection direction{AggregationDirection::ASC};
    /** @brief whether buckets with a null value sort first. */
    bool null_first{false};
};

/**
 * @brief A sort criterion of the top-hits aggregation.
 */
struct MILVUS_SDK_API AggregationSort {
    /**
     * @brief Constructor
     */
    AggregationSort() = default;

    /**
     * @brief Construct a top-hits sort criterion.
     *
     * @param [in] field_name field to sort by.
     * @param [in] direction sort direction.
     * @param [in] null_first whether documents with a null value sort first.
     */
    AggregationSort(std::string field_name, AggregationDirection direction, bool null_first = false);

    /** @brief field to sort by. */
    std::string field_name;
    /** @brief sort direction. */
    AggregationDirection direction{AggregationDirection::ASC};
    /** @brief whether documents with a null value sort first. */
    bool null_first{false};
};

/**
 * @brief Aggregation that returns the top hits per bucket, with an optional size and sort criteria.
 */
class MILVUS_SDK_API AggregationTopHits {
 public:
    /**
     * @brief Constructor
     */
    AggregationTopHits() = default;

    /**
     * @brief Construct a top-hits aggregation.
     *
     * @param [in] size number of top hits to return per bucket.
     */
    explicit AggregationTopHits(int64_t size);

    /**
     * @brief Get the number of top hits to return per bucket.
     * @return the size.
     */
    int64_t
    Size() const;

    /**
     * @brief Set the number of top hits to return per bucket.
     * @param [in] size the size.
     */
    void
    SetSize(int64_t size);

    /**
     * @brief Set the number of top hits to return per bucket.
     * @param [in] size the size.
     */
    AggregationTopHits&
    WithSize(int64_t size);

    /**
     * @brief Get the sort criteria of the top hits.
     * @return the sorts.
     */
    const std::vector<AggregationSort>&
    Sorts() const;

    /**
     * @brief Set the sort criteria of the top hits.
     * @param [in] sorts the sorts.
     */
    void
    SetSorts(std::vector<AggregationSort>&& sorts);

    /**
     * @brief Set the sort criteria of the top hits.
     * @param [in] sorts the sorts.
     */
    AggregationTopHits&
    WithSorts(std::vector<AggregationSort>&& sorts);

    /**
     * @brief Add a sort criterion to the top hits.
     * @param [in] sort the sort.
     */
    AggregationTopHits&
    AddSort(AggregationSort sort);

    /**
     * @brief Validate the aggregation configuration.
     * @return the validate.
     */
    Status
    Validate() const;

 private:
    int64_t size_{0};
    std::vector<AggregationSort> sorts_;
};

using AggregationTopHitsPtr = std::shared_ptr<AggregationTopHits>;

class SearchAggregation;
using SearchAggregationPtr = std::shared_ptr<SearchAggregation>;

/**
 * @brief A search aggregation over one or more group-by fields, with metrics, ordering, top hits and nesting.
 *
 * @par Used by
 * MilvusClientV2::Search() via SearchRequest::WithSearchAggregation().
 *
 * Not combinable with a non-zero offset, hybrid search, search/query iterators, a highlighter, or grouping search.
 * Supports at most four aggregation levels and at most ten bucket-key fields in total.
 */
class MILVUS_SDK_API SearchAggregation {
 public:
    /**
     * @brief Constructor
     */
    SearchAggregation() = default;

    /**
     * @brief Construct a search aggregation.
     *
     * @param [in] fields group-by field names.
     * @param [in] size number of buckets to return.
     */
    SearchAggregation(std::vector<std::string> fields, int64_t size);

    /**
     * @brief Get the group-by field names.
     * @return the fields.
     */
    const std::vector<std::string>&
    Fields() const;

    /**
     * @brief Set the group-by field names.
     * @param [in] fields the fields.
     */
    void
    SetFields(std::vector<std::string>&& fields);

    /**
     * @brief Set the group-by field names.
     * @param [in] fields the fields.
     */
    SearchAggregation&
    WithFields(std::vector<std::string>&& fields);

    /**
     * @brief Add a group-by field name.
     * @param [in] field the field.
     */
    SearchAggregation&
    AddField(std::string field);

    /**
     * @brief Get the number of buckets to return.
     * @return the size.
     */
    int64_t
    Size() const;

    /**
     * @brief Set the number of buckets to return.
     * @param [in] size the size.
     */
    void
    SetSize(int64_t size);

    /**
     * @brief Set the number of buckets to return.
     * @param [in] size the size.
     */
    SearchAggregation&
    WithSize(int64_t size);

    /**
     * @brief Get the aggregation metrics, keyed by alias.
     * @return the metrics.
     */
    const std::map<std::string, AggregationMetric>&
    Metrics() const;

    /**
     * @brief Set the aggregation metrics, keyed by alias.
     * @param [in] metrics the metrics.
     */
    void
    SetMetrics(std::map<std::string, AggregationMetric>&& metrics);

    /**
     * @brief Set the aggregation metrics, keyed by alias.
     * @param [in] metrics the metrics.
     */
    SearchAggregation&
    WithMetrics(std::map<std::string, AggregationMetric>&& metrics);

    /**
     * @brief Add an aggregation metric under an alias.
     * @param [in] alias the alias.
     * @param [in] metric the metric.
     */
    SearchAggregation&
    AddMetric(std::string alias, AggregationMetric metric);

    /**
     * @brief Get the bucket orderings.
     * @return the orders.
     */
    const std::vector<AggregationOrder>&
    Orders() const;

    /**
     * @brief Set the bucket orderings.
     * @param [in] orders the orders.
     */
    void
    SetOrders(std::vector<AggregationOrder>&& orders);

    /**
     * @brief Set the bucket orderings.
     * @param [in] orders the orders.
     */
    SearchAggregation&
    WithOrders(std::vector<AggregationOrder>&& orders);

    /**
     * @brief Add a bucket ordering.
     * @param [in] order the order.
     */
    SearchAggregation&
    AddOrder(AggregationOrder order);

    /**
     * @brief Get the top-hits aggregation of the buckets.
     * @return the top hits.
     */
    const AggregationTopHitsPtr&
    TopHits() const;

    /**
     * @brief Set the top-hits aggregation of the buckets.
     * @param [in] top_hits the top hits.
     */
    void
    SetTopHits(const AggregationTopHitsPtr& top_hits);

    /**
     * @brief Set the top-hits aggregation of the buckets.
     * @param [in] top_hits the top hits.
     */
    SearchAggregation&
    WithTopHits(const AggregationTopHitsPtr& top_hits);

    /**
     * @brief Get the nested sub-aggregation of the buckets.
     * @return the sub aggregation.
     */
    const SearchAggregationPtr&
    SubAggregation() const;

    /**
     * @brief Set the nested sub-aggregation of the buckets.
     * @param [in] sub_aggregation the sub aggregation.
     */
    void
    SetSubAggregation(const SearchAggregationPtr& sub_aggregation);

    /**
     * @brief Set the nested sub-aggregation of the buckets.
     * @param [in] sub_aggregation the sub aggregation.
     */
    SearchAggregation&
    WithSubAggregation(const SearchAggregationPtr& sub_aggregation);

    /**
     * @brief Validate the aggregation configuration.
     * @return the validate.
     */
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
