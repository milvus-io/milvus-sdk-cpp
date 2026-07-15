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

#include "milvus/request/dql/SearchRequest.h"

#include <memory>

#include "../../utils/Constants.h"
#include "../../utils/ExtraParamUtils.h"

namespace milvus {

SearchRequest&
SearchRequest::WithMetricType(::milvus::MetricType metric_type) {
    SetMetricType(metric_type);
    return *this;
}

SearchRequest&
SearchRequest::AddExtraParam(const std::string& key, const std::string& value) {
    SearchRequestBase::AddExtraParam(key, value);
    return *this;
}

SearchRequest&
SearchRequest::WithExtraParams(const std::unordered_map<std::string, std::string>& params) {
    for (const auto& pair : params) {
        SearchRequestBase::AddExtraParam(pair.first, pair.second);
    }
    return *this;
}

SearchRequest&
SearchRequest::WithLimit(int64_t limit) {
    SetLimit(limit);
    return *this;
}

SearchRequest&
SearchRequest::WithFilter(std::string filter) {
    SetFilter(std::move(filter));
    return *this;
}

SearchRequest&
SearchRequest::WithAnnsField(const std::string& ann_field) {
    SetAnnsField(ann_field);
    return *this;
}

SearchRequest&
SearchRequest::AddFilterTemplate(std::string key, const nlohmann::json& filter_template) {
    SearchRequestBase::AddFilterTemplate(key, filter_template);
    return *this;
}

SearchRequest&
SearchRequest::WithFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates) {
    SetFilterTemplates(std::move(filter_templates));
    return *this;
}

int64_t
SearchRequest::Offset() const {
    return GetExtraInt64(extra_params_, "offset", 0);
}

void
SearchRequest::SetOffset(int64_t offset) {
    SetExtraInt64(extra_params_, "offset", offset);
}

SearchRequest&
SearchRequest::WithOffset(int64_t offset) {
    SetOffset(offset);
    return *this;
}

int64_t
SearchRequest::RoundDecimal() const {
    return GetExtraInt64(extra_params_, "round_decimal", -1);
}

void
SearchRequest::SetRoundDecimal(int64_t round_decimal) {
    SetExtraInt64(extra_params_, "round_decimal", round_decimal);
}

SearchRequest&
SearchRequest::WithRoundDecimal(int64_t round_decimal) {
    SetRoundDecimal(round_decimal);
    return *this;
}

bool
SearchRequest::IgnoreGrowing() const {
    return GetExtraBool(extra_params_, "ignore_growing", false);
}

void
SearchRequest::SetIgnoreGrowing(bool ignore_growing) {
    SetExtraBool(extra_params_, "ignore_growing", ignore_growing);
}

SearchRequest&
SearchRequest::WithIgnoreGrowing(bool ignore_growing) {
    SetIgnoreGrowing(ignore_growing);
    return *this;
}

std::string
SearchRequest::GroupByField() const {
    return GetExtraStr(extra_params_, "group_by_field", "");
}

void
SearchRequest::SetGroupByField(const std::string& field_name) {
    SetExtraStr(extra_params_, "group_by_field", field_name);
}

SearchRequest&
SearchRequest::WithGroupByField(const std::string& field_name) {
    SetGroupByField(field_name);
    return *this;
}

int64_t
SearchRequest::GroupSize() const {
    return GetExtraInt64(extra_params_, "group_size", 1);
}

void
SearchRequest::SetGroupSize(int64_t group_size) {
    SetExtraInt64(extra_params_, "group_size", group_size);
}

SearchRequest&
SearchRequest::WithGroupSize(int64_t group_size) {
    SetGroupSize(group_size);
    return *this;
}

bool
SearchRequest::StrictGroupSize() const {
    return GetExtraBool(extra_params_, "strict_group_size", false);
}

void
SearchRequest::SetStrictGroupSize(bool strict_group_size) {
    SetExtraBool(extra_params_, "strict_group_size", strict_group_size);
}

SearchRequest&
SearchRequest::WithStrictGroupSize(bool strict_group_size) {
    SetStrictGroupSize(strict_group_size);
    return *this;
}

SearchRequest&
SearchRequest::WithRadius(double radius) {
    SetRadius(radius);
    return *this;
}

SearchRequest&
SearchRequest::WithRangeFilter(double filter) {
    SetRangeFilter(filter);
    return *this;
}

const FunctionScorePtr&
SearchRequest::Rerank() const {
    return ranker_;
}

void
SearchRequest::SetRerank(const FunctionScorePtr& ranker) {
    ranker_ = ranker;
}

SearchRequest&
SearchRequest::WithRerank(const FunctionScorePtr& ranker) {
    ranker_ = ranker;
    return *this;
}

SearchRequest&
SearchRequest::WithTimezone(const std::string& timezone) {
    SetTimezone(timezone);
    return *this;
}

const HighlighterPtr&
SearchRequest::GetHighlighter() const {
    return highlighter_;
}

void
SearchRequest::SetHighlighter(const HighlighterPtr& highlighter) {
    highlighter_ = highlighter;
}

SearchRequest&
SearchRequest::WithHighlighter(const HighlighterPtr& highlighter) {
    highlighter_ = highlighter;
    return *this;
}

const SearchAggregationPtr&
SearchRequest::GetSearchAggregation() const {
    return search_aggregation_;
}

void
SearchRequest::SetSearchAggregation(const SearchAggregationPtr& aggregation) {
    search_aggregation_ = aggregation;
}

SearchRequest&
SearchRequest::WithSearchAggregation(const SearchAggregationPtr& aggregation) {
    SetSearchAggregation(aggregation);
    return *this;
}

const std::vector<OrderByField>&
SearchRequest::OrderByFields() const {
    return order_by_fields_;
}

void
SearchRequest::SetOrderByFields(std::vector<OrderByField>&& order_by_fields) {
    order_by_fields_ = std::move(order_by_fields);
}

SearchRequest&
SearchRequest::WithOrderByFields(std::vector<OrderByField>&& order_by_fields) {
    SetOrderByFields(std::move(order_by_fields));
    return *this;
}

SearchRequest&
SearchRequest::AddOrderByField(OrderByField order_by_field) {
    order_by_fields_.emplace_back(std::move(order_by_field));
    return *this;
}

Status
SearchRequest::Validate() const {
    auto status = SearchRequestBase::Validate();
    if (!status.IsOk()) {
        return status;
    }
    if (search_aggregation_ == nullptr) {
        return Status::OK();
    }
    if (!GroupByField().empty()) {
        return {StatusCode::INVALID_ARGUMENT, "group_by_field and search_aggregation cannot be used simultaneously"};
    }
    return search_aggregation_->Validate();
}

}  // namespace milvus
