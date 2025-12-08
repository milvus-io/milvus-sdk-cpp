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

#include <vector>

#include "FieldSchema.h"

namespace milvus {

/**
 * @brief Struct field schema used by CollectionSchema
 */
class StructFieldSchema {
 public:
    /**
     * @brief Constructor
     */
    StructFieldSchema();

    /**
     * @brief Constructor
     */
    explicit StructFieldSchema(std::string name, std::string description = "");

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
    StructFieldSchema&
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
    StructFieldSchema&
    WithDescription(std::string description);

    /**
     * @brief Get max capacity for an array field
     */
    int64_t
    MaxCapacity() const;

    /**
     * @brief Quickly set max capacity for an array field
     */
    void
    SetMaxCapacity(int64_t capacity);

    /**
     * @brief Quickly set max capacity for an array field
     */
    StructFieldSchema&
    WithMaxCapacity(int64_t capacity);

    /**
     * @brief Fields schema array.
     */
    const std::vector<FieldSchema>&
    Fields() const;

    /**
     * @brief Add a field schema.
     */
    StructFieldSchema&
    AddField(const FieldSchema& field_schema);

    /**
     * @brief Add a field schema.
     */
    StructFieldSchema&
    AddField(FieldSchema&& field_schema);

 private:
    std::string name_;
    std::string description_;
    int64_t capacity_{0};

    std::vector<FieldSchema> fields_;
};
}  // namespace milvus
