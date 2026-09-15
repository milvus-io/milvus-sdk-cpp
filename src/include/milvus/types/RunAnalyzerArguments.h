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
#include <string>
#include <unordered_map>
#include <vector>

#include "../Status.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Arguments for MilvusClient::RunAnalyzer().
 */
class MILVUS_SDK_API RunAnalyzerArguments {
 public:
    /**
     * @brief Constructor
     */
    RunAnalyzerArguments() = default;
    virtual ~RunAnalyzerArguments() = default;

    /**
     * @brief Get the target db name.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set target db name, default is empty, means use the db name of MilvusClient.
     * @param [in] db_name the DB name.
     */
    Status
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set target db name, default is empty, means use the db name of MilvusClient.
     * @param [in] db_name the DB name.
     */
    RunAnalyzerArguments&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Get name of the target collection.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of this collection, cannot be empty.
     * @param [in] collection_name the collection name.
     */
    Status
    SetCollectionName(std::string collection_name);

    /**
     * @brief Set name of this collection, cannot be empty.
     * @param [in] collection_name the collection name.
     */
    RunAnalyzerArguments&
    WithCollectionName(std::string collection_name);

    /**
     * @brief Get name of the target field.
     * @return the field name.
     */
    const std::string&
    FieldName() const;

    /**
     * @brief Set name of the target field, cannot be empty.
     * @param [in] field_name the field name.
     */
    Status
    SetFieldName(std::string field_name);

    /**
     * @brief Set name of the target field, cannot be empty.
     * @param [in] field_name the field name.
     */
    RunAnalyzerArguments&
    WithFieldName(std::string field_name);

    /**
     * @brief Get texts to be analyzed.
     * @return the texts.
     */
    const std::vector<std::string>&
    Texts() const;

    /**
     * @brief Set texts to be analyzed.
     * @param [in] texts the texts.
     */
    Status
    SetTexts(const std::vector<std::string>& texts);

    /**
     * @brief Set texts to be analyzed.
     * @param [in] texts the texts.
     */
    RunAnalyzerArguments&
    WithTexts(const std::vector<std::string>& texts);

    /**
     * @brief Add text for analyze.
     * @param [in] text the text.
     */
    RunAnalyzerArguments&
    AddText(std::string text);

    /**
     * @brief Get analyzer names.
     * @return the analyzer names.
     */
    const std::vector<std::string>&
    AnalyzerNames() const;

    /**
     * @brief Set analyzer names.
     * @param [in] names the names.
     */
    Status
    SetAnalyzerNames(const std::vector<std::string>& names);

    /**
     * @brief Specify an analyzer.
     * @param [in] name the name.
     */
    RunAnalyzerArguments&
    AddAnalyzerName(std::string name);

    /**
     * @brief Get analyzer parameters.
     * @return the analyzer params.
     */
    const nlohmann::json&
    AnalyzerParams() const;

    /**
     * @brief Set analyzer parameters.
     * @param [in] params the params.
     */
    Status
    SetAnalyzerParams(const nlohmann::json& params);

    /**
     * @brief Set analyzer parameters.
     * @param [in] params the params.
     */
    RunAnalyzerArguments&
    WithAnalyzerParams(const nlohmann::json& params);

    /**
     * @brief Return details or not.
     * @return true if detail is included.
     */
    bool
    IsWithDetail() const;

    /**
     * @brief Include details in the results.
     * @param [in] with_detail the with detail.
     */
    RunAnalyzerArguments&
    WithDetail(bool with_detail);

    /**
     * @brief Return hash values or not.
     * @return true if the hash is included.
     */
    bool
    IsWithHash() const;

    /**
     * @brief Include hash values in the results.
     * @param [in] with_hash the with hash.
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
