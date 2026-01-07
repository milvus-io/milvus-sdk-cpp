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

#include "../../types/IDArray.h"
#include "./DMLRequestBase.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Delete()
 */
class DeleteRequest : public DMLRequestBase<DeleteRequest> {
 public:
    /**
     * @brief Constructor
     */
    DeleteRequest() = default;

    /**
     * @brief Get filter expression.
     */
    const std::string&
    Filter() const;

    /**
     * @brief Set filter expression.
     */
    void
    SetFilter(const std::string& filter);

    /**
     * @brief Set filter expression.
     */
    DeleteRequest&
    WithFilter(const std::string& filter);

    /**
     * @brief Get filter templates.
     */
    const std::unordered_map<std::string, nlohmann::json>&
    FilterTemplates() const;

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
    DeleteRequest&
    AddFilterTemplate(std::string key, nlohmann::json&& filter_template);

    /**
     * @brief Set filter templates. Only take effect when filter is not empty.
     */
    void
    SetFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates);

    /**
     * @brief Set filter templates. Only take effect when filter is not empty.
     */
    DeleteRequest&
    WithFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates);

    /**
     * @brief Add a filter template. Only take effect when filter is not empty.
     */
    DeleteRequest&
    AddFilterTemplate(const std::string& key, nlohmann::json&& filter_template);

    /**
     * @brief Get primary keys to be deleted. Only take effect when filter is empty.
     */
    const IDArray&
    IDs() const;

    /**
     * @brief Set integer primary keys to be deleted. Only take effect when filter is empty.
     */
    void
    SetIDs(std::vector<int64_t>&& id_array);

    /**
     * @brief Set integer primary keys to be deleted. Only take effect when filter is empty.
     */
    DeleteRequest&
    WithIDs(std::vector<int64_t>&& id_array);

    /**
     * @brief Set string primary keys to be deleted. Only take effect when filter is empty.
     */
    void
    SetIDs(std::vector<std::string>&& id_array);

    /**
     * @brief Set string primary keys to be deleted. Only take effect when filter is empty.
     */
    DeleteRequest&
    WithIDs(std::vector<std::string>&& id_array);

 private:
    std::string filter_;
    std::unordered_map<std::string, nlohmann::json> filter_templates_;
    IDArray ids_;
};

}  // namespace milvus
