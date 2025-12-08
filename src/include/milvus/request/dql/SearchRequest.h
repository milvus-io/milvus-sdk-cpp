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

#include "../../types/FunctionScore.h"
#include "../../types/SubSearchRequest.h"
#include "./DQLRequestBase.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Search()
 */
class SearchRequest : public DQLRequestBase, public SearchRequestBase {
 public:
    /**
     * @brief Constructor
     */
    SearchRequest() = default;

    /**
     * @brief Set database name in which the collection is created.
     */
    SearchRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Set name of the collection.
     */
    SearchRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Set the partition names.
     * If partition nemes are empty, will query in the entire collection.
     */
    SearchRequest&
    WithPartitionNames(std::set<std::string>&& partition_names);

    /**
     * @brief Add a partition name.
     */
    SearchRequest&
    AddPartitionName(const std::string& partition_name);

    /**
     * @brief Set the output field names.
     */
    SearchRequest&
    WithOutputFields(std::set<std::string>&& output_field_names);

    /**
     * @brief Add an output field.
     */
    SearchRequest&
    AddOutputField(const std::string& output_field);

    /**
     * @brief Set the consistency level.
     */
    SearchRequest&
    WithConsistencyLevel(ConsistencyLevel consistency_level);

    /**
     * @brief Specifies the metric type.
     */
    SearchRequest&
    WithMetricType(::milvus::MetricType metric_type);

    /**
     * @brief Set search limit(topk).
     * Note: this value is stored in the ExtraParams
     */
    SearchRequest&
    WithLimit(int64_t limit);

    /**
     * @brief Set filter expression.
     */
    SearchRequest&
    WithFilter(std::string filter);

    /**
     * @brief Add a filter template. Only take effect when filter is not empty.
     * Expression template, to improve expression parsing performance in complicated list.
     * Assume user has a filter = "pk > 3 and city in ["beijing", "shanghai", ......]
     * The long list of city will increase the time cost to parse this expression.
     * So, we provide filterTemplate for this purpose, user can set filter like this:
     *     filter = "pk > {age} and city in {city}"
     *     filterTemplate = {"age": 3, "city": ["beijing", "shanghai", ......]}
     * Valid value of a template can be:
     *     boolean, numeric, string, array
     */
    SearchRequest&
    AddFilterTemplate(std::string key, const nlohmann::json& filter_template);

    /**
     * @brief Set filter templates. Only take effect when filter is not empty.
     */
    SearchRequest&
    WithFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates);

    /**
     * @brief Get offset value.
     */
    int64_t
    Offset() const;

    /**
     * @brief Set offset value.
     * Note: this value is stored in the SubSearchRequest::ExtraParams.
     */
    void
    SetOffset(int64_t offset);

    /**
     * @brief Set offset value.
     * Note: this value is stored in the SubSearchRequest::ExtraParams.
     */
    SearchRequest&
    WithOffset(int64_t offset);

    /**
     * @brief Get round decimal value.
     */
    int64_t
    RoundDecimal() const;

    /**
     * @brief Set round decimal value.
     */
    void
    SetRoundDecimal(int64_t round_decimal);

    /**
     * @brief Set round decimal value.
     */
    SearchRequest&
    WithRoundDecimal(int64_t round_decimal);

    /**
     * @brief Get ignore growing flag.
     */
    bool
    IgnoreGrowing() const;

    /**
     * @brief Set ignore growing flag.
     */
    void
    SetIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Set ignore growing flag.
     */
    SearchRequest&
    WithIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Get group by field value.
     */
    std::string
    GroupByField() const;

    /**
     * @brief Set group by field value.
     */
    void
    SetGroupByField(const std::string& field_name);

    /**
     * @brief Set group by field value.
     */
    SearchRequest&
    WithGroupByField(const std::string& field_name);

    /**
     * @brief Get group size value.
     */
    int64_t
    GroupSize() const;

    /**
     * @brief Set group size value.
     */
    void
    SetGroupSize(int64_t group_size);

    /**
     * @brief Set group size value.
     */
    SearchRequest&
    WithGroupSize(int64_t group_size);

    /**
     * @brief Get strict group size flag.
     */
    bool
    StrictGroupSize() const;

    /**
     * @brief Set strict group size flag.
     */
    void
    SetStrictGroupSize(bool strict_group_size);

    /**
     * @brief Set strict group size flag.
     */
    SearchRequest&
    WithStrictGroupSize(bool strict_group_size);

    /**
     * @brief Get reranker
     *
     */
    const FunctionScorePtr&
    Rerank() const;

    /**
     * @brief Set reranker.
     * Allows multiple rerank functions such as Boost/Decay/Model, etc
     */
    void
    SetRerank(const FunctionScorePtr& ranker);

    /**
     * @brief Set reranker.
     * Allows multiple rerank functions such as Boost/Decay/Model, etc
     */
    SearchRequest&
    WithRerank(const FunctionScorePtr& ranker);

 private:
    FunctionScorePtr ranker_;
};

}  // namespace milvus
