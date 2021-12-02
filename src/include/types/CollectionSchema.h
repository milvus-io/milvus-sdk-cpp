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
    explicit CollectionSchema(const std::string& name, const std::string& desc = "", int32_t shard_num = 2)
        : name_(name), description_(desc), shard_num_(shard_num) {
    }

    const std::string&
    Name() const {
        return name_;
    }

    const std::string&
    Description() const {
        return description_;
    }

    const std::vector<FieldSchema>&
    Fields() const {
        return fields_;
    }

    bool
    AddField(FieldSchema& field_schema) {
        // TODO: check duplicate field name
        fields_.emplace_back(field_schema);
        return true;
    }

 private:
    /**
     * @brief Name of this collection, cannot be empty
     */
    std::string name_;

    /**
     * @brief Description of this collection, can be empty
     */
    std::string description_;

    /**
     * @brief Set shards number, the number must be larger than zero, default value is 2.
     */
    int32_t shard_num_ = 2;

    /**
     * @brief Schema for each field.
     */
    std::vector<FieldSchema> fields_;
};

}  // namespace milvus
