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

#include <vector>

#include "../../types/FunctionChain.h"
#include "../../types/FunctionScore.h"
#include "../../types/Highlighter.h"
#include "../../types/IDArray.h"
#include "../../types/OrderByField.h"
#include "../../types/SearchAggregation.h"
#include "../../types/SearchRequestBase.h"
#include "./DQLRequestBase.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Search()
 * @par Example
 * @code
 * milvus::SearchResponse response;
 * milvus::SearchRequest request;
 * request.WithCollectionName("demo").WithLimit(3).WithOutputFields({"id"});
 * request.AddFloatVector(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f});
 * if (client->Search(request, response).IsOk()) {
 *     auto& ids = response.Results().Ids();
 *     auto top_ids = ids.IntIDArray();
 * }
 * @endcode
 */
class MILVUS_SDK_API SearchRequest : public DQLRequestBase<SearchRequest>,
                                     public SearchRequestVectorAssigner<SearchRequest> {
 public:
    /**
     * @brief Constructor
     */
    SearchRequest() = default;

    /**
     * @brief Get primary keys whose vectors are used as search targets.
     * @return the i ds.
     */
    const IDArray&
    IDs() const;

    /**
     * @brief Set integer primary keys whose vectors are used as search targets.
     * Note: IDs and target vectors cannot be specified at the same time.
     * @param [in] id_array the ID array.
     */
    void
    SetIDs(std::vector<int64_t>&& id_array);

    /**
     * @brief Set string primary keys whose vectors are used as search targets.
     * Note: IDs and target vectors cannot be specified at the same time.
     * @param [in] id_array the ID array.
     */
    void
    SetIDs(std::vector<std::string>&& id_array);

    /**
     * @brief Set integer primary keys whose vectors are used as search targets.
     * Note: IDs and target vectors cannot be specified at the same time.
     * @param [in] id_array the ID array.
     */
    SearchRequest&
    WithIDs(std::vector<int64_t>&& id_array);

    /**
     * @brief Set string primary keys whose vectors are used as search targets.
     * Note: IDs and target vectors cannot be specified at the same time.
     * @param [in] id_array the ID array.
     */
    SearchRequest&
    WithIDs(std::vector<std::string>&& id_array);

    /**
     * @brief Specifies the metric type.
     * @param [in] metric_type the metric type.
     */
    SearchRequest&
    WithMetricType(::milvus::MetricType metric_type);

    /**
     * @brief Add extra parameters such as "nlist", "ef".
     * @param [in] key the key.
     * @param [in] value the value.
     */
    SearchRequest&
    AddExtraParam(const std::string& key, const std::string& value);

    /**
     * @brief Add extra parameters such as "nlist", "ef".
     * @param [in] params the params.
     */
    SearchRequest&
    WithExtraParams(const std::unordered_map<std::string, std::string>& params);

    /**
     * @brief Set search limit(topk).
     * Note: this value is stored in the ExtraParams.
     * @param [in] limit the limit.
     */
    SearchRequest&
    WithLimit(int64_t limit);

    /**
     * @brief Set filter expression.
     * @param [in] filter the filter.
     */
    SearchRequest&
    WithFilter(std::string filter);

    /**
     * @brief Set target field of ann search.
     * @param [in] ann_field the ANN field.
     */
    SearchRequest&
    WithAnnsField(const std::string& ann_field);

    /**
     * @brief Add a filter template. Only take effect when filter is not empty.
     * Expression template, to improve expression parsing performance in complicated list.
     * Assume user has a filter = "pk > 3 and city in ["beijing", "shanghai", ......]
     * The long list of city will increase the time cost to parse this expression.
     * So, we provide filterTemplate for this purpose, user can set filter like this:
     *     filter = "pk > {age} and city in {city}"
     *     filterTemplate = {"age": 3, "city": ["beijing", "shanghai", ......]}
     * Valid value of a template can be:
     *     boolean, numeric, string, array.
     *
     * Read the doc for more info: https://milvus.io/docs/filtering-templating.md#Filter-Templating
     * @param [in] key the key.
     * @param [in] filter_template the filter template.
     */
    SearchRequest&
    AddFilterTemplate(std::string key, const nlohmann::json& filter_template);

    /**
     * @brief Set filter templates. Only take effect when filter is not empty.
     * Read the doc for more info: https://milvus.io/docs/filtering-templating.md#Filter-Templating
     * @param [in] filter_templates the filter templates.
     */
    SearchRequest&
    WithFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates);

    /**
     * @brief Get offset value.
     * @return the offset.
     */
    int64_t
    Offset() const;

    /**
     * @brief Set offset value.
     * Note: this value is stored in the ExtraParams.
     * @param [in] offset the offset.
     */
    void
    SetOffset(int64_t offset);

    /**
     * @brief Set offset value.
     * Note: this value is stored in the ExtraParams.
     * @param [in] offset the offset.
     */
    SearchRequest&
    WithOffset(int64_t offset);

    /**
     * @brief Get round decimal value.
     * @return the round decimal.
     */
    int64_t
    RoundDecimal() const;

    /**
     * @brief Set round decimal value.
     * @param [in] round_decimal the round decimal.
     */
    void
    SetRoundDecimal(int64_t round_decimal);

    /**
     * @brief Set round decimal value.
     * @param [in] round_decimal the round decimal.
     */
    SearchRequest&
    WithRoundDecimal(int64_t round_decimal);

    /**
     * @brief Get ignore growing flag.
     * @return the ignore growing.
     */
    bool
    IgnoreGrowing() const;

    /**
     * @brief Set ignore growing flag.
     * @param [in] ignore_growing the ignore growing.
     */
    void
    SetIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Set ignore growing flag.
     * @param [in] ignore_growing the ignore growing.
     */
    SearchRequest&
    WithIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Get group by field value.
     * @return the group by field.
     */
    std::string
    GroupByField() const;

    /**
     * @brief Set group by field value.
     * @param [in] field_name the field name.
     */
    void
    SetGroupByField(const std::string& field_name);

    /**
     * @brief Set group by field value.
     * @param [in] field_name the field name.
     */
    SearchRequest&
    WithGroupByField(const std::string& field_name);

    /**
     * @brief Get group size value.
     * @return the group size.
     */
    int64_t
    GroupSize() const;

    /**
     * @brief Set group size value.
     * @param [in] group_size the group size.
     */
    void
    SetGroupSize(int64_t group_size);

    /**
     * @brief Set group size value.
     * @param [in] group_size the group size.
     */
    SearchRequest&
    WithGroupSize(int64_t group_size);

    /**
     * @brief Get strict group size flag.
     * @return the strict group size.
     */
    bool
    StrictGroupSize() const;

    /**
     * @brief Set strict group size flag.
     * @param [in] strict_group_size the strict group size.
     */
    void
    SetStrictGroupSize(bool strict_group_size);

    /**
     * @brief Set strict group size flag.
     * @param [in] strict_group_size the strict group size.
     */
    SearchRequest&
    WithStrictGroupSize(bool strict_group_size);

    /**
     * @brief Set range radius.
     * Note: this value is stored in the ExtraParams.
     * @param [in] radius the radius.
     */
    SearchRequest&
    WithRadius(double radius);

    /**
     * @brief Set range filter.
     * Note: this value is stored in the ExtraParams.
     * @param [in] filter the filter.
     */
    SearchRequest&
    WithRangeFilter(double filter);

    /**
     * @brief Get reranker.
     *
     * @return the rerank.
     */
    const FunctionScorePtr&
    Rerank() const;

    /**
     * @brief Set reranker.
     * Allows multiple rerank functions such as Boost/Decay/Model, etc.
     * Read the doc for more info: https://milvus.io/docs/boost-ranker.md
     * @param [in] ranker the ranker.
     */
    void
    SetRerank(const FunctionScorePtr& ranker);

    /**
     * @brief Set reranker.
     * Allows multiple rerank functions such as Boost/Decay/Model, etc.
     * Read the doc for more info: https://milvus.io/docs/boost-ranker.md
     * @param [in] ranker the ranker.
     */
    SearchRequest&
    WithRerank(const FunctionScorePtr& ranker);

    /**
     * @brief Get function chains.
     * @return the function chains.
     */
    const std::vector<FunctionChain>&
    FunctionChains() const;

    /**
     * @brief Set function chains. Function chains and rerank cannot be used together.
     * @param [in] function_chains the function chains.
     */
    void
    SetFunctionChains(std::vector<FunctionChain>&& function_chains);

    /**
     * @brief Set function chains. Function chains and rerank cannot be used together.
     * @param [in] function_chains the function chains.
     */
    SearchRequest&
    WithFunctionChains(std::vector<FunctionChain>&& function_chains);

    /**
     * @brief Add a function chain. Function chains and rerank cannot be used together.
     * @param [in] function_chain the function chain.
     */
    SearchRequest&
    AddFunctionChain(const FunctionChain& function_chain);

    /**
     * @brief Set timezone, takes effect for Timestamptz field.
     * Read the doc for more info:
     * https://milvus.io/docs/single-vector-search.md#Temporarily-set-a-timezone-for-a-search
     * @param [in] timezone the timezone.
     */
    SearchRequest&
    WithTimezone(const std::string& timezone);

    /**
     * @brief Get highlighter.
     * @return the highlighter.
     */
    const HighlighterPtr&
    GetHighlighter() const;

    /**
     * @brief Set highlighter.
     * @param [in] highlighter the highlighter.
     */
    void
    SetHighlighter(const HighlighterPtr& highlighter);

    /**
     * @brief Set highlighter.
     * @param [in] highlighter the highlighter.
     */
    SearchRequest&
    WithHighlighter(const HighlighterPtr& highlighter);

    /**
     * @brief Get search aggregation settings.
     * @return the search aggregation.
     */
    const SearchAggregationPtr&
    GetSearchAggregation() const;

    /**
     * @brief Set search aggregation settings.
     * @param [in] aggregation the aggregation.
     */
    void
    SetSearchAggregation(const SearchAggregationPtr& aggregation);

    /**
     * @brief Set search aggregation settings.
     * @param [in] aggregation the aggregation.
     */
    SearchRequest&
    WithSearchAggregation(const SearchAggregationPtr& aggregation);

    /**
     * @brief Get fields used to order search results.
     * @return the order by fields.
     */
    const std::vector<OrderByField>&
    OrderByFields() const;

    /**
     * @brief Set fields used to order search results.
     * @param [in] order_by_fields the order by fields.
     */
    void
    SetOrderByFields(std::vector<OrderByField>&& order_by_fields);

    /**
     * @brief Set fields used to order search results.
     * @param [in] order_by_fields the order by fields.
     */
    SearchRequest&
    WithOrderByFields(std::vector<OrderByField>&& order_by_fields);

    /**
     * @brief Add a field used to order search results.
     * @param [in] order_by_field the order by field.
     */
    SearchRequest&
    AddOrderByField(OrderByField order_by_field);

    Status
    Validate() const;

 private:
    IDArray ids_;
    FunctionScorePtr ranker_;
    HighlighterPtr highlighter_;
    SearchAggregationPtr search_aggregation_;
    std::vector<OrderByField> order_by_fields_;
    std::vector<FunctionChain> function_chains_;
};

}  // namespace milvus
