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
#include <memory>
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <string>
#include <unordered_set>
#include <vector>

#include "FieldSchema.h"
#include "Function.h"
#include "StructFieldSchema.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Collection schema for MilvusClient::CreateCollection().
 * @par Example
 * @code
 * milvus::CollectionSchema schema("demo");
 * schema.SetDescription("demo collection");
 * schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "", true, false));
 * schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR, "").WithDimension(128));
 * auto status = client->CreateCollection(milvus::CreateCollectionRequest()
 *                                            .WithCollectionName("demo")
 *                                            .WithCollectionSchema(std::make_shared<milvus::CollectionSchema>(schema)));
 * @endcode
 */
class MILVUS_SDK_API CollectionSchema {
 public:
    /**
     * @brief Constructor
     */
    CollectionSchema();

    /**
     * @brief Constructor
     */
    explicit CollectionSchema(std::string name, std::string desc = "", int32_t shard_num = 1,
                              bool enable_dynamic_field = true);

    /**
     * @brief Collection name, cannot be empty.
     * @deprecated in MilvusClientV2, collection name is passed by CreateCollectionRequest.
     * @return the name.
     */
    const std::string&
    Name() const;

    /**
     * @brief Set collection name, cannot be empty.
     * @deprecated in MilvusClientV2, collection name is passed by CreateCollectionRequest.
     * @param [in] name the name.
     */
    void
    SetName(std::string name);

    /**
     * @brief Collection description, can be empty.
     * @deprecated in MilvusClientV2, description is passed by CreateCollectionRequest.
     * @return the description.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set collection description, can be empty.
     * @deprecated in MilvusClientV2, description is passed by CreateCollectionRequest.
     * @param [in] description the description.
     */
    void
    SetDescription(std::string description);

    /**
     * @brief Collection shards number, the number must be larger than zero, default value is 2.
     * @deprecated in MilvusClientV2, shardsNum is passed by CreateCollectionRequest.
     * @return the shards num.
     */
    int32_t
    ShardsNum() const;

    /**
     * @brief Set shards number, the number must be larger than zero, default value is 2.
     * @deprecated in MilvusClientV2, shardsNum is passed by CreateCollectionRequest.
     * @param [in] num the num.
     */
    void
    SetShardsNum(int32_t num);

    /**
     * @brief Whether undeclared fields are stored in the hidden $meta dynamic field.
     *
     * When enabled, any field not declared in the schema is stored as a key-value pair in a
     * hidden JSON field named $meta.
     * @return the enable dynamic field.
     */
    bool
    EnableDynamicField() const;

    /**
     * @brief Enable or disable the dynamic field.
     *
     * When enabled, undeclared insert fields are stored in the hidden $meta JSON field.
     * @param [in] enable_dynamic_field the enable dynamic field.
     */
    void
    SetEnableDynamicField(bool enable_dynamic_field);

    /**
     * @brief Fields schema array.
     * @return the fields.
     */
    const std::vector<FieldSchema>&
    Fields() const;

    /**
     * @brief Add a field schema.
     * @param [in] field_schema the field schema.
     */
    bool
    AddField(const FieldSchema& field_schema);

    /**
     * @brief Add a field schema.
     * @param [in] field_schema the field schema.
     */
    bool
    AddField(FieldSchema&& field_schema);

    /**
     * @brief Struct fields schema array.
     * @return the struct fields.
     */
    const std::vector<StructFieldSchema>&
    StructFields() const;

    /**
     * @brief Add a struct field schema.
     * @param [in] field_schema the field schema.
     */
    bool
    AddStructField(const StructFieldSchema& field_schema);

    /**
     * @brief Add a struct field schema.
     * @param [in] field_schema the field schema.
     */
    bool
    AddStructField(StructFieldSchema&& field_schema);

    /**
     * @brief Return Anns field names.
     * @return the anns field names.
     */
    std::unordered_set<std::string>
    AnnsFieldNames() const;

    /**
     * @brief Return the primary key field name.
     * @return the primary field name.
     */
    std::string
    PrimaryFieldName() const;

    /**
     * @brief Get functions array.
     * @return the functions.
     */
    const std::vector<FunctionPtr>&
    Functions() const;

    /**
     * @brief Add a function.
     */
    void
    AddFunction(const FunctionPtr& function);

    /**
     * @brief Get external collection source path.
     * @return the external source.
     */
    const std::string&
    ExternalSource() const;

    /**
     * @brief Set external collection source path.
     * @param [in] external_source the external source.
     */
    void
    SetExternalSource(std::string external_source);

    /**
     * @brief Set external collection source path.
     * @param [in] external_source the external source.
     */
    CollectionSchema&
    WithExternalSource(std::string external_source);

    /**
     * @brief Get external collection spec JSON.
     * @return the external spec.
     */
    const nlohmann::json&
    ExternalSpec() const;

    /**
     * @brief Set external collection spec JSON.
     * @param [in] external_spec the external spec.
     */
    void
    SetExternalSpec(const nlohmann::json& external_spec);

    /**
     * @brief Set external collection spec JSON.
     * @param [in] external_spec the external spec.
     */
    CollectionSchema&
    WithExternalSpec(const nlohmann::json& external_spec);

 private:
    std::string name_;
    std::string description_;
    int32_t shard_num_ = 1;  // from v2.4, the default shard_num is 1(old version is 2)
    bool enable_dynamic_field_{true};
    std::vector<FieldSchema> fields_;
    std::vector<StructFieldSchema> struct_fields_;

    std::vector<FunctionPtr> functions_;
    std::string external_source_;
    nlohmann::json external_spec_;
};

using CollectionSchemaPtr = std::shared_ptr<CollectionSchema>;

}  // namespace milvus
