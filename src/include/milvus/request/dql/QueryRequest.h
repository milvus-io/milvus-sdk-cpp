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

#include <milvus/thirdparty/nlohmann/json.hpp>
#include <unordered_map>
#include <vector>

#include "./DQLRequestBase.h"
#include "milvus/Export.h"
#include "milvus/types/IDArray.h"
#include "milvus/types/OrderByField.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Query()
 * @par Example
 * @code
 * milvus::QueryResponse response;
 * milvus::QueryRequest request;
 * request.WithCollectionName("demo").WithFilter("id >= 100").WithOutputFields({"id"});
 * if (client->Query(request, response).IsOk()) {
 *     milvus::EntityRow row;
 *     for (int i = 0; i < response.Results().GetRowCount(); ++i) {
 *         response.Results().OutputRow(i, row);
 *     }
 * }
 * @endcode
 */
class MILVUS_SDK_API QueryRequest : public DQLRequestBase<QueryRequest> {
 public:
    /**
     * @brief Constructor
     */
    QueryRequest() = default;

    /**
     * @brief Get id array.
     * @return the i ds.
     */
    const IDArray&
    IDs() const;

    /**
     * @brief Set integer IDs to query.
     * Note: IDs and filter cannot be set at the same time.
     * @param [in] id_array the ID array.
     */
    void
    SetIDs(std::vector<int64_t>&& id_array);

    /**
     * @brief Set string IDs to query.
     * Note: IDs and filter cannot be set at the same time.
     * @param [in] id_array the ID array.
     */
    void
    SetIDs(std::vector<std::string>&& id_array);

    /**
     * @brief Set integer IDs to query.
     * Note: IDs and filter cannot be set at the same time.
     * @param [in] id_array the ID array.
     */
    QueryRequest&
    WithIDs(std::vector<int64_t>&& id_array);

    /**
     * @brief Set string IDs to query.
     * Note: IDs and filter cannot be set at the same time.
     * @param [in] id_array the ID array.
     */
    QueryRequest&
    WithIDs(std::vector<std::string>&& id_array);

    /**
     * @brief Get filter expression.
     * @return the filter.
     */
    const std::string&
    Filter() const;

    /**
     * @brief Set filter expression.
     * @param [in] filter the filter.
     */
    void
    SetFilter(std::string filter);

    /**
     * @brief Set filter expression.
     * @param [in] filter the filter.
     */
    QueryRequest&
    WithFilter(std::string filter);

    /**
     * @brief Get filter templates.
     * @return the filter templates.
     */
    const std::unordered_map<std::string, nlohmann::json>&
    FilterTemplates() const;

    /**
     * @brief Set filter templates.
     * Read the doc for more info: https://milvus.io/docs/filtering-templating.md#Filter-Templating
     * @param [in] filter_templates the filter templates.
     */
    void
    SetFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates);

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
     * Read the doc for more info: https://milvus.io/docs/filtering-templating.md#Filter-Templating
     * @param [in] key the key.
     * @param [in] filter_template the filter template.
     */
    QueryRequest&
    AddFilterTemplate(std::string key, const nlohmann::json& filter_template);

    /**
     * @brief Set filter templates. Only take effect when filter is not empty.
     * Read the doc for more info: https://milvus.io/docs/filtering-templating.md#Filter-Templating
     * @param [in] filter_templates the filter templates.
     */
    QueryRequest&
    WithFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates);

    /**
     * @brief Get limit value.
     * @return the limit.
     */
    int64_t
    Limit() const;

    /**
     * @brief Set limit value, only avaiable when expression is empty. \n
     * Note: this value is stored in the ExtraParams.
     * @param [in] limit the limit.
     */
    void
    SetLimit(int64_t limit);

    /**
     * @brief Set limit value, only avaiable when expression is empty. \n
     * Note: this value is stored in the ExtraParams.
     * @param [in] limit the limit.
     */
    QueryRequest&
    WithLimit(int64_t limit);

    /**
     * @brief Get offset value.
     * @return the offset.
     */
    int64_t
    Offset() const;

    /**
     * @brief Set offset value, only avaiable when expression is empty. \n
     * Note: this value is stored in the ExtraParams.
     * @param [in] offset the offset.
     */
    void
    SetOffset(int64_t offset);

    /**
     * @brief Set offset value, only avaiable when expression is empty. \n
     * Note: this value is stored in the ExtraParams.
     * @param [in] offset the offset.
     */
    QueryRequest&
    WithOffset(int64_t offset);

    /**
     * @brief Get ignore growing segments.
     * @return the ignore growing.
     */
    bool
    IgnoreGrowing() const;

    /**
     * @brief Set ignore growing segments.
     * Note: this value is stored in the ExtraParams.
     * @param [in] ignore_growing the ignore growing.
     */
    void
    SetIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Set ignore growing segments.
     * Note: this value is stored in the ExtraParams.
     * @param [in] ignore_growing the ignore growing.
     */
    QueryRequest&
    WithIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Add extra param.
     * @param [in] key the key.
     * @param [in] value the value.
     */
    QueryRequest&
    AddExtraParam(const std::string& key, const std::string& value);

    /**
     * @brief Get extra param.
     * @return the extra params.
     */
    const std::unordered_map<std::string, std::string>&
    ExtraParams() const;

    /**
     * @brief Get timezone, takes effect for Timestamptz field.
     * Note: this value is stored in the ExtraParams.
     * @return the timezone.
     */
    std::string
    Timezone() const;

    /**
     * @brief Set timezone, takes effect for Timestamptz field.
     * Note: this value is stored in the ExtraParams.
     * @param [in] timezone the timezone.
     */
    void
    SetTimezone(const std::string& timezone);

    /**
     * @brief Set timezone, takes effect for Timestamptz field.
     * Note: this value is stored in the ExtraParams.
     * @param [in] timezone the timezone.
     */
    QueryRequest&
    WithTimezone(const std::string& timezone);

    /**
     * @brief Get fields used to order query results.
     * @return the order by fields.
     */
    const std::vector<OrderByField>&
    OrderByFields() const;

    /**
     * @brief Set fields used to order query results.
     * @param [in] order_by_fields the order by fields.
     */
    void
    SetOrderByFields(std::vector<OrderByField>&& order_by_fields);

    /**
     * @brief Set fields used to order query results.
     * @param [in] order_by_fields the order by fields.
     */
    QueryRequest&
    WithOrderByFields(std::vector<OrderByField>&& order_by_fields);

    /**
     * @brief Add a field used to order query results.
     * @param [in] order_by_field the order by field.
     */
    QueryRequest&
    AddOrderByField(OrderByField order_by_field);

 private:
    IDArray ids_;
    std::string filter_;
    std::unordered_map<std::string, nlohmann::json> filter_templates_;
    std::unordered_map<std::string, std::string> extra_params_;
    std::vector<OrderByField> order_by_fields_;
};

}  // namespace milvus
