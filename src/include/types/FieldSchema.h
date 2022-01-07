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

#include <map>
#include <string>

#include "types/Constants.h"
#include "types/DataType.h"

namespace milvus {
class FieldSchema {
 public:
    FieldSchema() = default;

    FieldSchema(const std::string& name, DataType data_type, const std::string& description = "",
                bool is_primary_key = false, bool auto_id = false)
        : name_(name),
          description_(description),
          data_type_(data_type),
          is_primary_key_(is_primary_key),
          auto_id_(auto_id) {
    }

    /**
     * @brief Name of this field, cannot be empty.
     */
    const std::string&
    Name() const {
        return name_;
    }

    /**
     * @brief Set name of the field.
     */
    void
    SetName(const std::string& name) {
        name_ = name;
    }

    /**
     * @brief Description of this field, can be empty.
     */
    const std::string&
    Description() const {
        return description_;
    }

    /**
     * @brief Set description of the field.
     */
    void
    SetDescription(const std::string& description) {
        description_ = description;
    }

    /**
     * @brief Field data type.
     */
    DataType
    FieldDataType() const {
        return data_type_;
    }

    /**
     * @brief Set field data type.
     */
    void
    SetDataType(const DataType dt) {
        data_type_ = dt;
    }

    /**
     * @brief The field is primary key or not.
     *
     * Each collection only has one primary key.
     * Currently only int64 type field can be primary key .
     */
    bool
    IsPrimaryKey() const {
        return is_primary_key_;
    }

    /**
     * @brief Set field to be primary key.
     */
    void
    SetPrimaryKey(bool is_primary_key) {
        is_primary_key_ = is_primary_key;
    }

    /**
     * @brief Field item's id is auto-generated or not.
     *
     * If ths flag is true, server will generate id when data is inserted.
     * Else the client must provide id for each entity when insert data.
     */
    bool
    AutoID() const {
        return auto_id_;
    }

    /**
     * @brief Set field item's id to be auto-generated.
     */
    void
    SetAutoID(bool auto_id) {
        auto_id_ = auto_id;
    }

    /**
     * @brief Extra key-value pair setting for this field
     *
     * Currently vector field need to input "dim":"x" to specify dimension.
     */
    const std::map<std::string, std::string>&
    TypeParams() const {
        return type_params_;
    }

    /**
     * @brief Set extra key-value pair setting for this field
     *
     * Currently vector field need to input "dim":"x" to specify dimension.
     */
    void
    SetTypeParams(std::map<std::string, std::string>&& params) {
        type_params_ = std::move(params);
    }

    /**
     * @brief Quickly set dimension for a vector field
     */
    bool
    SetDimension(uint32_t dimension) {
        if (dimension == 0) {
            return false;
        }

        type_params_.insert(std::make_pair(FieldDim(), std::to_string(dimension)));
        return true;
    }

 private:
    std::string name_;
    std::string description_;
    DataType data_type_;
    bool is_primary_key_ = false;
    bool auto_id_ = false;
    std::map<std::string, std::string> type_params_;
};
}  // namespace milvus
