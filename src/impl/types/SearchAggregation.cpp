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

#include "milvus/types/SearchAggregation.h"

#include <set>
#include <utility>

namespace milvus {

AggregationMetric::AggregationMetric(AggregationMetricOp op, std::string field_name)
    : op(op), field_name(std::move(field_name)) {
}

AggregationOrder::AggregationOrder(std::string key, AggregationDirection direction, bool null_first)
    : key(std::move(key)), direction(direction), null_first(null_first) {
}

AggregationSort::AggregationSort(std::string field_name, AggregationDirection direction, bool null_first)
    : field_name(std::move(field_name)), direction(direction), null_first(null_first) {
}

AggregationTopHits::AggregationTopHits(int64_t size) : size_(size) {
}

int64_t
AggregationTopHits::Size() const {
    return size_;
}

void
AggregationTopHits::SetSize(int64_t size) {
    size_ = size;
}

AggregationTopHits&
AggregationTopHits::WithSize(int64_t size) {
    SetSize(size);
    return *this;
}

const std::vector<AggregationSort>&
AggregationTopHits::Sorts() const {
    return sorts_;
}

void
AggregationTopHits::SetSorts(std::vector<AggregationSort>&& sorts) {
    sorts_ = std::move(sorts);
}

AggregationTopHits&
AggregationTopHits::WithSorts(std::vector<AggregationSort>&& sorts) {
    SetSorts(std::move(sorts));
    return *this;
}

AggregationTopHits&
AggregationTopHits::AddSort(AggregationSort sort) {
    sorts_.emplace_back(std::move(sort));
    return *this;
}

Status
AggregationTopHits::Validate() const {
    if (size_ <= 0) {
        return {StatusCode::INVALID_ARGUMENT, "AggregationTopHits size must be positive"};
    }
    for (const auto& sort : sorts_) {
        if (sort.field_name.empty()) {
            return {StatusCode::INVALID_ARGUMENT, "AggregationTopHits sort field name cannot be empty"};
        }
    }
    return Status::OK();
}

SearchAggregation::SearchAggregation(std::vector<std::string> fields, int64_t size)
    : fields_(std::move(fields)), size_(size) {
}

const std::vector<std::string>&
SearchAggregation::Fields() const {
    return fields_;
}

void
SearchAggregation::SetFields(std::vector<std::string>&& fields) {
    fields_ = std::move(fields);
}

SearchAggregation&
SearchAggregation::WithFields(std::vector<std::string>&& fields) {
    SetFields(std::move(fields));
    return *this;
}

SearchAggregation&
SearchAggregation::AddField(std::string field) {
    fields_.emplace_back(std::move(field));
    return *this;
}

int64_t
SearchAggregation::Size() const {
    return size_;
}

void
SearchAggregation::SetSize(int64_t size) {
    size_ = size;
}

SearchAggregation&
SearchAggregation::WithSize(int64_t size) {
    SetSize(size);
    return *this;
}

const std::map<std::string, AggregationMetric>&
SearchAggregation::Metrics() const {
    return metrics_;
}

void
SearchAggregation::SetMetrics(std::map<std::string, AggregationMetric>&& metrics) {
    metrics_ = std::move(metrics);
}

SearchAggregation&
SearchAggregation::WithMetrics(std::map<std::string, AggregationMetric>&& metrics) {
    SetMetrics(std::move(metrics));
    return *this;
}

SearchAggregation&
SearchAggregation::AddMetric(std::string alias, AggregationMetric metric) {
    metrics_[std::move(alias)] = std::move(metric);
    return *this;
}

const std::vector<AggregationOrder>&
SearchAggregation::Orders() const {
    return orders_;
}

void
SearchAggregation::SetOrders(std::vector<AggregationOrder>&& orders) {
    orders_ = std::move(orders);
}

SearchAggregation&
SearchAggregation::WithOrders(std::vector<AggregationOrder>&& orders) {
    SetOrders(std::move(orders));
    return *this;
}

SearchAggregation&
SearchAggregation::AddOrder(AggregationOrder order) {
    orders_.emplace_back(std::move(order));
    return *this;
}

const AggregationTopHitsPtr&
SearchAggregation::TopHits() const {
    return top_hits_;
}

void
SearchAggregation::SetTopHits(const AggregationTopHitsPtr& top_hits) {
    top_hits_ = top_hits;
}

SearchAggregation&
SearchAggregation::WithTopHits(const AggregationTopHitsPtr& top_hits) {
    SetTopHits(top_hits);
    return *this;
}

const SearchAggregationPtr&
SearchAggregation::SubAggregation() const {
    return sub_aggregation_;
}

void
SearchAggregation::SetSubAggregation(const SearchAggregationPtr& sub_aggregation) {
    sub_aggregation_ = sub_aggregation;
}

SearchAggregation&
SearchAggregation::WithSubAggregation(const SearchAggregationPtr& sub_aggregation) {
    SetSubAggregation(sub_aggregation);
    return *this;
}

Status
SearchAggregation::Validate() const {
    std::set<const SearchAggregation*> aggregation_chain;
    auto current = this;
    while (current != nullptr) {
        if (!aggregation_chain.insert(current).second) {
            return {StatusCode::INVALID_ARGUMENT, "SearchAggregation cannot contain a cycle"};
        }
        current = current->SubAggregation().get();
    }

    if (fields_.empty()) {
        return {StatusCode::INVALID_ARGUMENT, "SearchAggregation fields cannot be empty"};
    }
    for (const auto& field : fields_) {
        if (field.empty()) {
            return {StatusCode::INVALID_ARGUMENT, "SearchAggregation field cannot be empty"};
        }
    }
    if (size_ <= 0) {
        return {StatusCode::INVALID_ARGUMENT, "SearchAggregation size must be positive"};
    }
    for (const auto& metric : metrics_) {
        if (metric.first.empty()) {
            return {StatusCode::INVALID_ARGUMENT, "SearchAggregation metric alias cannot be empty"};
        }
        if (metric.second.field_name.empty()) {
            return {StatusCode::INVALID_ARGUMENT, "SearchAggregation metric field name cannot be empty"};
        }
        if (metric.second.op != AggregationMetricOp::COUNT && metric.second.field_name == "*") {
            return {StatusCode::INVALID_ARGUMENT, "Only count aggregation supports the '*' field"};
        }
    }
    static const std::set<std::string> special_order_keys{"_count", "_key"};
    for (const auto& order : orders_) {
        if (order.key.empty()) {
            return {StatusCode::INVALID_ARGUMENT, "SearchAggregation order key cannot be empty"};
        }
        if (metrics_.find(order.key) == metrics_.end() &&
            special_order_keys.find(order.key) == special_order_keys.end()) {
            return {StatusCode::INVALID_ARGUMENT,
                    "SearchAggregation order key must be a metric alias or one of _count/_key"};
        }
    }
    if (top_hits_ != nullptr) {
        auto status = top_hits_->Validate();
        if (!status.IsOk()) {
            return status;
        }
    }
    if (sub_aggregation_ != nullptr) {
        auto status = sub_aggregation_->Validate();
        if (!status.IsOk()) {
            return status;
        }
    }
    return Status::OK();
}

}  // namespace milvus
