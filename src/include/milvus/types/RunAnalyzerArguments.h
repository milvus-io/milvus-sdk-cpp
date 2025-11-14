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
#include <string>
#include <unordered_map>
#include <vector>

#include "../Status.h"

namespace milvus {

/**
 * @brief Arguments for MilvusClient::RunAnalyzer().
 */
class RunAnalyzerArguments {
 public:
    RunAnalyzerArguments() = default;
    virtual ~RunAnalyzerArguments() = default;

    /**
     * @brief Get the target db name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set target db name, default is empty, means use the db name of MilvusClient.
     */
    Status
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set target db name, default is empty, means use the db name of MilvusClient.
     */
    RunAnalyzerArguments&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Get name of the target collection.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of this collection, cannot be empty.
     */
    Status
    SetCollectionName(std::string collection_name);

    /**
     * @brief Set name of this collection, cannot be empty.
     */
    RunAnalyzerArguments&
    WithCollectionName(std::string collection_name);

    /**
     * @brief Get name of the target field.
     */
    const std::string&
    FieldName() const;

    /**
     * @brief Set name of the target field, cannot be empty.
     */
    Status
    SetFieldName(std::string field_name);

    /**
     * @brief Set name of the target field, cannot be empty.
     */
    RunAnalyzerArguments&
    WithFieldName(std::string field_name);

    /**
     * @brief Get texts to be analyzed.
     */
    const std::vector<std::string>&
    Texts() const;

    /**
     * @brief Set texts to be analyzed.
     */
    Status
    SetTexts(const std::vector<std::string>& texts);

    /**
     * @brief Add text for analyze.
     */
    RunAnalyzerArguments&
    AddText(std::string text);

    /**
     * @brief Get analyzer names.
     */
    const std::vector<std::string>&
    AnalyzerNames() const;

    /**
     * @brief Set analyzer names.
     */
    Status
    SetAnalyzerNames(const std::vector<std::string>& names);

    /**
     * @brief Specify an analyzer.
     */
    RunAnalyzerArguments&
    AddAnalyzerName(std::string name);

    /**
     * @brief Get analyzer parameters.
     */
    const nlohmann::json&
    AnalyzerParams() const;

    /**
     * @brief Set analyzer parameters.
     */
    Status
    SetAnalyzerParams(const nlohmann::json& params);

    /**
     * @brief Set analyzer parameters.
     */
    RunAnalyzerArguments&
    WithAnalyzerParams(const nlohmann::json& params);

    /**
     * @brief Return details or not.
     */
    bool
    IsWithDetail() const;

    /**
     * @brief Include details in the results.
     */
    RunAnalyzerArguments&
    WithDetail(bool with_detail);

    /**
     * @brief Return hash values or not.
     */
    bool
    IsWithHash() const;

    /**
     * @brief Include hash values in the results.
     */
    RunAnalyzerArguments&
    WithHash(bool with_hash);

 private:
    std::string db_name_;
    std::string collection_name_;
    std::string field_name_;

    std::vector<std::string> texts_;
    std::vector<std::string> analyzer_names_;
    nlohmann::json analyzer_params_;
    bool with_detail_{false};
    bool with_hash_{false};
};

}  // namespace milvus
