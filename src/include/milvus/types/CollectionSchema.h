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
#include <string>
#include <unordered_set>
#include <vector>

#include "FieldSchema.h"
#include "Function.h"

namespace milvus {

/**
 * @brief Collection schema for MilvusClient::CreateCollection().
 */
class CollectionSchema {
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
     */
    const std::string&
    Name() const;

    /**
     * @brief Set collection name, cannot be empty.
     * @deprecated in MilvusClientV2, collection name is passed by CreateCollectionRequest.
     */
    void
    SetName(std::string name);

    /**
     * @brief Collection description, can be empty.
     * @deprecated in MilvusClientV2, description is passed by CreateCollectionRequest.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set collection description, can be empty.
     * @deprecated in MilvusClientV2, description is passed by CreateCollectionRequest.
     */
    void
    SetDescription(std::string description);

    /**
     * @brief Collection shards number, the number must be larger than zero, default value is 2.
     * @deprecated in MilvusClientV2, shardsNum is passed by CreateCollectionRequest.
     */
    int32_t
    ShardsNum() const;

    /**
     * @brief Set shards number, the number must be larger than zero, default value is 2.
     * @deprecated in MilvusClientV2, shardsNum is passed by CreateCollectionRequest.
     */
    void
    SetShardsNum(int32_t num);

    bool
    EnableDynamicField() const;

    void
    SetEnableDynamicField(bool enable_dynamic_field);

    /**
     * @brief Fields schema array.
     */
    const std::vector<FieldSchema>&
    Fields() const;

    /**
     * @brief Add a field schema.
     */
    bool
    AddField(const FieldSchema& field_schema);

    /**
     * @brief Add a field schema.
     */
    bool
    AddField(FieldSchema&& field_schema);

    /**
     * @brief Return Anns field names.
     */
    std::unordered_set<std::string>
    AnnsFieldNames() const;

    /**
     * @brief Return the primary key field name.
     */
    std::string
    PrimaryFieldName() const;

    /**
     * @brief Get functions array.
     */
    const std::vector<FunctionPtr>&
    Functions() const;

    /**
     * @brief Add a function.
     */
    void
    AddFunction(const FunctionPtr& function);

 private:
    std::string name_;
    std::string description_;
    int32_t shard_num_ = 1;  // from v2.4, the default shard_num is 1(old version is 2)
    bool enable_dynamic_field_{true};
    std::vector<FieldSchema> fields_;

    std::vector<FunctionPtr> functions_;
};

using CollectionSchemaPtr = std::shared_ptr<CollectionSchema>;

}  // namespace milvus
