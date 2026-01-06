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
#include <nlohmann/json.hpp>
#include <string>

#include "../Status.h"
#include "DataType.h"

namespace milvus {

/**
 * @brief Field schema used by CollectionSchema
 */
class FieldSchema {
 public:
    FieldSchema();

    /**
     * @brief Constructor
     */
    FieldSchema(std::string name, DataType data_type, std::string description = "", bool is_primary_key = false,
                bool auto_id = false);

    /**
     * @brief Name of this field, cannot be empty.
     */
    const std::string&
    Name() const;

    /**
     * @brief Set name of the field.
     */
    void
    SetName(std::string name);

    /**
     * @brief Set name of the field.
     */
    FieldSchema&
    WithName(std::string name);

    /**
     * @brief Description of this field, can be empty.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set description of the field.
     */
    void
    SetDescription(std::string description);

    /**
     * @brief Set description of the field.
     */
    FieldSchema&
    WithDescription(std::string description);

    /**
     * @brief Field data type.
     */
    DataType
    FieldDataType() const;

    /**
     * @brief Set field data type.
     */
    void
    SetDataType(DataType dt);

    /**
     * @brief Set field data type.
     */
    FieldSchema&
    WithDataType(DataType dt);

    /**
     * @brief Element type of array field.
     */
    DataType
    ElementType() const;

    /**
     * @brief Set element type for array field.
     */
    void
    SetElementType(DataType dt);

    /**
     * @brief Set element type for array field.
     */
    FieldSchema&
    WithElementType(DataType dt);

    /**
     * @brief The field is primary key or not.
     *
     * Each collection only has one primary key.
     * Currently only int64 type field can be primary key.
     */
    bool
    IsPrimaryKey() const;

    /**
     * @brief Set field to be primary key.
     */
    void
    SetPrimaryKey(bool is_primary_key);

    /**
     * @brief Set field to be primary key.
     */
    FieldSchema&
    WithPrimaryKey(bool is_primary_key);

    /**
     * @brief Field item's id is auto-generated or not.
     *
     * If ths flag is true, server will generate id when data is inserted.
     * Else the client must provide id for each entity when insert data.
     */
    bool
    AutoID() const;

    /**
     * @brief Set field item's id to be auto-generated.
     */
    void
    SetAutoID(bool auto_id);

    /**
     * @brief Set field item's id to be auto-generated.
     */
    FieldSchema&
    WithAutoID(bool auto_id);

    /**
     * @brief Field item's id is partition key or not.
     *
     */
    bool
    IsPartitionKey() const;

    /**
     * @brief Set field item's id to be partition key.
     */
    void
    SetPartitionKey(bool partition_key);

    /**
     * @brief Set field item's id to be partition key.
     */
    FieldSchema&
    WithPartitionKey(bool partition_key);

    /**
     * @brief Field item's id is clustering key or not.
     *
     */
    bool
    IsClusteringKey() const;

    /**
     * @brief Set field item's id to be clustering key.
     */
    void
    SetClusteringKey(bool clustering_key);

    /**
     * @brief Set field item's id to be clustering key.
     */
    FieldSchema&
    WithClusteringKey(bool clustering_key);

    /**
     * @brief Extra key-value pair setting for this field.
     */
    const std::map<std::string, std::string>&
    TypeParams() const;

    /**
     * @brief Set extra key-value pair setting for this field.
     * Note: the values inputted by SetDimension/SetMaxLength/SetMaxCapacity are stored in typeParams as a map.
     */
    void
    SetTypeParams(const std::map<std::string, std::string>& params);

    /**
     * @brief Set extra key-value pair setting for this field
     * Note: the values inputted by SetDimension/SetMaxLength/SetMaxCapacity are stored in typeParams as a map.
     */
    void
    SetTypeParams(std::map<std::string, std::string>&& params);

    /**
     * @brief Add an extra key-value pair setting for this field
     */
    FieldSchema&
    AddTypeParam(const std::string& key, const std::string& val);

    /**
     * @brief Get dimension for a vector field.
     */
    uint32_t
    Dimension() const;

    /**
     * @brief Quickly set dimension for a vector field.
     */
    bool
    SetDimension(uint32_t dimension);

    /**
     * @brief Quickly set dimension for a vector field.
     */
    FieldSchema&
    WithDimension(uint32_t dimension);

    /**
     * @brief Get max length for a varchar field.
     */
    uint32_t
    MaxLength() const;

    /**
     * @brief Quickly set max length for a varchar field.
     */
    void
    SetMaxLength(uint32_t length);

    /**
     * @brief Quickly set max length for a varchar field.
     */
    FieldSchema&
    WithMaxLength(uint32_t length);

    /**
     * @brief Get max capacity of an array field.
     */
    uint32_t
    MaxCapacity() const;

    /**
     * @brief Quickly set max capacity for an array field.
     */
    void
    SetMaxCapacity(uint32_t capacity);

    /**
     * @brief Quickly set max capacity for an array field.
     */
    FieldSchema&
    WithMaxCapacity(uint32_t capacity);

    /**
     * @brief Enable enable text analysis/tokenize for varchar field.
     */
    FieldSchema&
    EnableAnalyzer(bool enableAnalyzer);

    /**
     * @brief Get the flag whether enable analyzer.
     */
    bool
    IsEnableAnalyzer() const;

    /**
     * @brief Enable text match for varchar field.
     */
    FieldSchema&
    EnableMatch(bool enableMatch);

    /**
     * @brief Get the flag whether enable text match.
     */
    bool
    IsEnableMatch() const;

    /**
     * @brief Set analyzer parameters.
     * Note: AnalyzerParams and MultiAnalyzerParams cannot be applied on the same field.
     * Read the doc for more into: https://milvus.io/docs/analyzer-overview.md
     */
    void
    SetAnalyzerParams(const nlohmann::json& params);

    /**
     * @brief Set analyzer parameters.
     * Note: AnalyzerParams and MultiAnalyzerParams cannot be applied on the same field.
     * Read the doc for more into: https://milvus.io/docs/analyzer-overview.md
     */
    FieldSchema&
    WithAnalyzerParams(const nlohmann::json& params);

    /**
     * @brief Get analyzer parameters.
     */
    nlohmann::json
    AnalyzerParams() const;

    /**
     * @brief Set multi analyzer parameters.
     * Note: AnalyzerParams and MultiAnalyzerParams cannot be applied on the same field.
     * Read the doc for more info: https://milvus.io/docs/multi-language-analyzers.md
     */
    void
    SetMultiAnalyzerParams(const nlohmann::json& params);

    /**
     * @brief Set multi analyzer parameters.
     * Note: AnalyzerParams and MultiAnalyzerParams cannot be applied on the same field.
     * Read the doc for more info: https://milvus.io/docs/multi-language-analyzers.md
     */
    FieldSchema&
    WithMultiAnalyzerParams(const nlohmann::json& params);

    /**
     * @brief Get multi analyzer parameters.
     */
    nlohmann::json
    MultiAnalyzerParams() const;

    /**
     * @brief Get the flag whether the field value is nullable.
     */
    bool
    IsNullable() const;

    /**
     * @brief Set field value can be nullable or not.
     *
     * Note: all scalar fields, excluding the primary field, support nullable.
     */
    void
    SetNullable(bool nullable);

    /**
     * @brief Set field value can be nullable or not.
     *
     * Note: all scalar fields, excluding the primary field, support nullable.
     */
    FieldSchema&
    WithNullable(bool nullable);

    /**
     * @brief Set default value of this field.
     *
     * Note: JSON and Array fields do not support default values.
     * @param [in] val only accept JSON primitive types.
     */
    void
    SetDefaultValue(const nlohmann::json& val);

    /**
     * @brief Set default value of this field.
     *
     * Note: JSON and Array fields do not support default values.
     * @param [in] val only accept JSON primitive types.
     */
    FieldSchema&
    WithDefaultValue(const nlohmann::json& val);

    /**
     * @brief Get default value of this field.
     */
    const nlohmann::json&
    DefaultValue() const;

 private:
    std::string name_;
    std::string description_;
    DataType data_type_{DataType::UNKNOWN};
    DataType element_type_{DataType::UNKNOWN};  // only for array field
    bool is_primary_key_ = false;
    bool auto_id_ = false;
    bool is_partition_key_ = false;
    bool is_clustering_key_ = false;
    std::map<std::string, std::string> type_params_;

    bool is_nullable_ = false;
    nlohmann::json default_value_;  // only accept primitive types
};
}  // namespace milvus
