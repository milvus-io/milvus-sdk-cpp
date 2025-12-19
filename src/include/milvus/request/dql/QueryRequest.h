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

#include <nlohmann/json.hpp>
#include <unordered_map>

#include "./DQLRequestBase.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Query()
 */
class QueryRequest : public DQLRequestBase {
 public:
    /**
     * @brief Constructor
     */
    QueryRequest() = default;

    /**
     * @brief Set database name in which the collection is created.
     */
    QueryRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Set name of the collection.
     */
    QueryRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Set the partition names
     * If partition nemes are empty, will query in the entire collection.
     */
    QueryRequest&
    WithPartitionNames(std::set<std::string>&& partition_names);

    /**
     * @brief Add a partition name.
     */
    QueryRequest&
    AddPartitionName(const std::string& partition_name);

    /**
     * @brief Set the output field names
     */
    QueryRequest&
    WithOutputFields(std::set<std::string>&& output_field_names);

    /**
     * @brief Add an output field.
     */
    QueryRequest&
    AddOutputField(const std::string& output_field);

    /**
     * @brief Set the consistency level
     */
    QueryRequest&
    WithConsistencyLevel(ConsistencyLevel consistency_level);

    /**
     * @brief Get filter expression.
     */
    const std::string&
    Filter() const;

    /**
     * @brief Set filter expression.
     */
    void
    SetFilter(std::string filter);

    /**
     * @brief Set filter expression
     */
    QueryRequest&
    WithFilter(std::string filter);

    /**
     * @brief Get filter templates
     */
    const std::unordered_map<std::string, nlohmann::json>&
    FilterTemplates() const;

    /**
     * @brief Set filter templates
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
     *     boolean, numeric, string, array
     */
    QueryRequest&
    AddFilterTemplate(std::string key, const nlohmann::json& filter_template);

    /**
     * @brief Set filter templates. Only take effect when filter is not empty.
     */
    QueryRequest&
    WithFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates);

    /**
     * @brief Get limit value.
     */
    int64_t
    Limit() const;

    /**
     * @brief Set limit value, only avaiable when expression is empty. \n
     * Note: this value is stored in the ExtraParams
     */
    void
    SetLimit(int64_t limit);

    /**
     * @brief Set limit value, only avaiable when expression is empty. \n
     * Note: this value is stored in the ExtraParams
     */
    QueryRequest&
    WithLimit(int64_t limit);

    /**
     * @brief Get offset value.
     */
    int64_t
    Offset() const;

    /**
     * @brief Set offset value, only avaiable when expression is empty. \n
     * Note: this value is stored in the ExtraParams
     */
    void
    SetOffset(int64_t offset);

    /**
     * @brief Set offset value, only avaiable when expression is empty. \n
     * Note: this value is stored in the ExtraParams
     */
    QueryRequest&
    WithOffset(int64_t offset);

    /**
     * @brief Get ignore growing segments.
     */
    bool
    IgnoreGrowing() const;

    /**
     * @brief Set ignore growing segments.
     * Note: this value is stored in the ExtraParams
     */
    void
    SetIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Set ignore growing segments.
     * Note: this value is stored in the ExtraParams
     */
    QueryRequest&
    WithIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Add extra param
     */
    QueryRequest&
    AddExtraParam(const std::string& key, const std::string& value);

    /**
     * @brief Get extra param
     */
    const std::unordered_map<std::string, std::string>&
    ExtraParams() const;

    /**
     * @brief Get timezone, takes effect for Timestamptz field.
     * Note: this value is stored in the ExtraParams
     */
    std::string
    Timezone() const;

    /**
     * @brief Set timezone, takes effect for Timestamptz field.
     * Note: this value is stored in the ExtraParams
     */
    void
    SetTimezone(const std::string& timezone);

    /**
     * @brief Set timezone, takes effect for Timestamptz field.
     * Note: this value is stored in the ExtraParams
     */
    QueryRequest&
    WithTimezone(const std::string& timezone);

 private:
    std::string filter_;
    std::unordered_map<std::string, nlohmann::json> filter_templates_;
    std::unordered_map<std::string, std::string> extra_params_;
};

}  // namespace milvus
