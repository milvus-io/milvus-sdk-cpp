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

#include <string>
#include <vector>

#include "FieldSchema.h"

namespace milvus {

/**
 * @brief Collection schema for CreateCollection().
 */
class CollectionSchema {
 public:
    CollectionSchema() = default;

    explicit CollectionSchema(const std::string& name, const std::string& desc = "", int32_t shard_num = 2)
        : name_(name), description_(desc), shard_num_(shard_num) {
    }

    /**
     * @brief Collection name, cannot be empty.
     */
    const std::string&
    Name() const {
        return name_;
    }

    /**
     * @brief Set collection name, cannot be empty.
     */
    void
    SetName(const std::string& name) {
        name_ = name;
    }

    /**
     * @brief Collection description, can be empty.
     */
    const std::string&
    Description() const {
        return description_;
    }

    /**
     * @brief Set collection description, can be empty.
     */
    void
    SetDescription(const std::string& description) {
        description_ = description;
    }

    /**
     * @brief Collection shards number, the number must be larger than zero, default value is 2.
     */
    int32_t
    ShardsNum() const {
        return shard_num_;
    }

    /**
     * @brief Set shards number, the number must be larger than zero, default value is 2.
     */
    void
    SetShardsNum(int32_t num) {
        shard_num_ = num;
    }

    /**
     * @brief Fields schema array.
     */
    const std::vector<FieldSchema>&
    Fields() const {
        return fields_;
    }

    /**
     * @brief Add a field schema.
     */
    bool
    AddField(const FieldSchema& field_schema) {
        // TODO: check duplicate field name
        fields_.emplace_back(field_schema);
        return true;
    }

    bool
    AddField(FieldSchema&& field_schema) {
        // TODO: check duplicate field name
        fields_.emplace_back(std::move(field_schema));
        return true;
    }

 private:
    std::string name_;
    std::string description_;
    int32_t shard_num_ = 2;
    std::vector<FieldSchema> fields_;
};

}  // namespace milvus
