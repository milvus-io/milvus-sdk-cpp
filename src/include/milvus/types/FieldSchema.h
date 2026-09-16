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
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <string>

#include "../Status.h"
#include "DataType.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Field schema used by CollectionSchema
 */
class MILVUS_SDK_API FieldSchema {
 public:
    FieldSchema();

    /**
     * @brief Constructor
     */
    FieldSchema(std::string name, DataType data_type, std::string description = "", bool is_primary_key = false,
                bool auto_id = false);

    /**
     * @brief Name of this field, cannot be empty.
     * @return the name.
     */
    const std::string&
    Name() const;

    /**
     * @brief Set name of the field.
     * @param [in] name the name.
     */
    void
    SetName(std::string name);

    /**
     * @brief Set name of the field.
     * @param [in] name the name.
     */
    FieldSchema&
    WithName(std::string name);

    /**
     * @brief Description of this field, can be empty.
     * @return the description.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set description of the field.
     * @param [in] description the description.
     */
    void
    SetDescription(std::string description);

    /**
     * @brief Set description of the field.
     * @param [in] description the description.
     */
    FieldSchema&
    WithDescription(std::string description);

    /**
     * @brief Field data type.
     * @return the field data type.
     */
    DataType
    FieldDataType() const;

    /**
     * @brief Set field data type.
     * @param [in] dt the dt.
     */
    void
    SetDataType(DataType dt);

    /**
     * @brief Set field data type.
     * @param [in] dt the dt.
     */
    FieldSchema&
    WithDataType(DataType dt);

    /**
     * @brief Element type of array field.
     * @return the element type.
     */
    DataType
    ElementType() const;

    /**
     * @brief Set element type for array field.
     * @param [in] dt the dt.
     */
    void
    SetElementType(DataType dt);

    /**
     * @brief Set element type for array field.
     * @param [in] dt the dt.
     */
    FieldSchema&
    WithElementType(DataType dt);

    /**
     * @brief The field is primary key or not.
     *
     * Each collection has exactly one primary key. Only INT64 and VARCHAR fields can be primary keys.
     * @return true if this field is the primary key.
     */
    bool
    IsPrimaryKey() const;

    /**
     * @brief Set field to be primary key.
     * @param [in] is_primary_key the is primary key.
     */
    void
    SetPrimaryKey(bool is_primary_key);

    /**
     * @brief Set field to be primary key.
     * @param [in] is_primary_key the is primary key.
     */
    FieldSchema&
    WithPrimaryKey(bool is_primary_key);

    /**
     * @brief Field item's id is auto-generated or not.
     *
     * Only applies to the primary key field. If true, the server generates IDs on insert.
     * Otherwise the client must provide an ID for each entity.
     * @return the auto ID.
     */
    bool
    AutoID() const;

    /**
     * @brief Set field item's id to be auto-generated.
     * @param [in] auto_id the auto ID.
     */
    void
    SetAutoID(bool auto_id);

    /**
     * @brief Set field item's id to be auto-generated.
     * @param [in] auto_id the auto ID.
     */
    FieldSchema&
    WithAutoID(bool auto_id);

    /**
     * @brief Field item's id is partition key or not.
     *
     * A partition key routes each entity to a partition. Partition key fields cannot be nullable.
     * @return true if this field is a partition key.
     */
    bool
    IsPartitionKey() const;

    /**
     * @brief Set field item's id to be partition key.
     *
     * Partition key fields cannot be nullable.
     * @param [in] partition_key the partition key.
     */
    void
    SetPartitionKey(bool partition_key);

    /**
     * @brief Set field item's id to be partition key.
     *
     * Partition key fields cannot be nullable.
     * @param [in] partition_key the partition key.
     */
    FieldSchema&
    WithPartitionKey(bool partition_key);

    /**
     * @brief Field item's id is clustering key or not.
     *
     * @return true if this field is a clustering key.
     */
    bool
    IsClusteringKey() const;

    /**
     * @brief Set field item's id to be clustering key.
     * @param [in] clustering_key the clustering key.
     */
    void
    SetClusteringKey(bool clustering_key);

    /**
     * @brief Set field item's id to be clustering key.
     * @param [in] clustering_key the clustering key.
     */
    FieldSchema&
    WithClusteringKey(bool clustering_key);

    /**
     * @brief Extra key-value pair setting for this field.
     * @return the type params.
     */
    const std::map<std::string, std::string>&
    TypeParams() const;

    /**
     * @brief Set extra key-value pair setting for this field.
     * Note: the values inputted by SetDimension/SetMaxLength/SetMaxCapacity are stored in typeParams as a map.
     * @param [in] params the params.
     */
    void
    SetTypeParams(const std::map<std::string, std::string>& params);

    /**
     * @brief Set extra key-value pair setting for this field
     * Note: the values inputted by SetDimension/SetMaxLength/SetMaxCapacity are stored in typeParams as a map.
     * @param [in] params the params.
     */
    void
    SetTypeParams(std::map<std::string, std::string>&& params);

    /**
     * @brief Add an extra key-value pair setting for this field
     * @param [in] key the key.
     * @param [in] val the val.
     */
    FieldSchema&
    AddTypeParam(const std::string& key, const std::string& val);

    /**
     * @brief Get dimension for a vector field.
     * @return the dimension.
     */
    uint32_t
    Dimension() const;

    /**
     * @brief Quickly set dimension for a vector field.
     * @param [in] dimension the dimension.
     */
    bool
    SetDimension(uint32_t dimension);

    /**
     * @brief Quickly set dimension for a vector field.
     * @param [in] dimension the dimension.
     */
    FieldSchema&
    WithDimension(uint32_t dimension);

    /**
     * @brief Get max length for a varchar field.
     * @return the max length.
     */
    uint32_t
    MaxLength() const;

    /**
     * @brief Quickly set max length for a varchar field.
     * @param [in] length the length.
     */
    void
    SetMaxLength(uint32_t length);

    /**
     * @brief Quickly set max length for a varchar field.
     * @param [in] length the length.
     */
    FieldSchema&
    WithMaxLength(uint32_t length);

    /**
     * @brief Get max capacity of an array field.
     * @return the max capacity.
     */
    uint32_t
    MaxCapacity() const;

    /**
     * @brief Quickly set max capacity for an array field.
     * @param [in] capacity the capacity.
     */
    void
    SetMaxCapacity(uint32_t capacity);

    /**
     * @brief Quickly set max capacity for an array field.
     * @param [in] capacity the capacity.
     */
    FieldSchema&
    WithMaxCapacity(uint32_t capacity);

    /**
     * @brief Enable enable text analysis/tokenize for varchar field.
     * @param [in] enableAnalyzer the enable analyzer.
     */
    FieldSchema&
    EnableAnalyzer(bool enableAnalyzer);

    /**
     * @brief Get the flag whether enable analyzer.
     * @return true if the analyzer is enabled.
     */
    bool
    IsEnableAnalyzer() const;

    /**
     * @brief Enable text match for varchar field.
     * @param [in] enableMatch the enable match.
     */
    FieldSchema&
    EnableMatch(bool enableMatch);

    /**
     * @brief Get the flag whether enable text match.
     * @return true if match is enabled.
     */
    bool
    IsEnableMatch() const;

    /**
     * @brief Set analyzer parameters.
     * Note: AnalyzerParams and MultiAnalyzerParams cannot be applied on the same field.
     * Read the doc for more into: https://milvus.io/docs/analyzer-overview.md
     * @param [in] params the params.
     */
    void
    SetAnalyzerParams(const nlohmann::json& params);

    /**
     * @brief Set analyzer parameters.
     * Note: AnalyzerParams and MultiAnalyzerParams cannot be applied on the same field.
     * Read the doc for more into: https://milvus.io/docs/analyzer-overview.md
     * @param [in] params the params.
     */
    FieldSchema&
    WithAnalyzerParams(const nlohmann::json& params);

    /**
     * @brief Get analyzer parameters.
     * @return the analyzer params.
     */
    nlohmann::json
    AnalyzerParams() const;

    /**
     * @brief Set multi analyzer parameters.
     * Note: AnalyzerParams and MultiAnalyzerParams cannot be applied on the same field.
     * Read the doc for more info: https://milvus.io/docs/multi-language-analyzers.md
     * @param [in] params the params.
     */
    void
    SetMultiAnalyzerParams(const nlohmann::json& params);

    /**
     * @brief Set multi analyzer parameters.
     * Note: AnalyzerParams and MultiAnalyzerParams cannot be applied on the same field.
     * Read the doc for more info: https://milvus.io/docs/multi-language-analyzers.md
     * @param [in] params the params.
     */
    FieldSchema&
    WithMultiAnalyzerParams(const nlohmann::json& params);

    /**
     * @brief Get multi analyzer parameters.
     * @return the multi analyzer params.
     */
    nlohmann::json
    MultiAnalyzerParams() const;

    /**
     * @brief Get the flag whether the field value is nullable.
     *
     * A nullable field stores NULL when its value is omitted or explicitly NULL on insert.
     * Primary and partition key fields cannot be nullable; vector fields with NULL cannot be
     * filtered by IS NULL expressions.
     * @return true if the field is nullable.
     */
    bool
    IsNullable() const;

    /**
     * @brief Set field value can be nullable or not.
     *
     * Scalar and vector fields (excluding the primary key) support nullable. Nullable fields
     * cannot be used as partition keys.
     * @param [in] nullable the nullable.
     */
    void
    SetNullable(bool nullable);

    /**
     * @brief Set field value can be nullable or not.
     *
     * Scalar and vector fields (excluding the primary key) support nullable. Nullable fields
     * cannot be used as partition keys.
     * @param [in] nullable the nullable.
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
     * @return the default value.
     */
    const nlohmann::json&
    DefaultValue() const;

    /**
     * @brief Get external field mapping name.
     * @return the external field.
     */
    const std::string&
    ExternalField() const;

    /**
     * @brief Set external field mapping name.
     * @param [in] external_field the external field.
     */
    void
    SetExternalField(std::string external_field);

    /**
     * @brief Set external field mapping name.
     * @param [in] external_field the external field.
     */
    FieldSchema&
    WithExternalField(std::string external_field);

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
    std::string external_field_;
};
}  // namespace milvus
