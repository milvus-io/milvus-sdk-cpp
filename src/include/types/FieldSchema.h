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

#include "types/DataType.h"

namespace milvus {
class FieldSchema {
 public:
    FieldSchema(const std::string& name, DataType data_type, const std::string& description = "",
                bool is_primary_key = false, bool auto_id = false)
        : name_(name),
          description_(description),
          data_type_(data_type),
          is_primary_key_(is_primary_key),
          auto_id_(auto_id) {
    }

    const std::string&
    name() const {
        return name_;
    }

    const std::string&
    description() const {
        return description_;
    }

    DataType
    data_type() const {
        return data_type_;
    }

    bool
    is_primary_key() const {
        return is_primary_key_;
    }

    bool
    auto_id() const {
        return auto_id_;
    }

    const std::map<std::string, std::string>&
    type_params() const {
        return type_params_;
    }

    bool
    SetDimension(uint32_t dimension) {
        if (dimension == 0) {
            return false;
        }

        type_params_.insert(std::make_pair("dim", std::to_string(dimension)));
        return true;
    }

 private:
    /**
     * @brief Name of this field, cannot be empty
     */
    std::string name_;

    /**
     * @brief Description of this field, can be empty
     */
    std::string description_;

    /**
     * @brief Field data tpye
     */
    DataType data_type_;

    /**
     * @brief Specify the field to be primary key
     *
     * Each collection only has one primary key.
     * Currently only int64 type field can be primary key .
     */
    bool is_primary_key_ = false;

    /**
     * @brief Let server automatically generate id for this field
     *
     * If ths flag is true, server will generate id when data is inserted.
     * Else the client must provide id for each entity when insert data.
     */
    bool auto_id_ = false;

    /**
     * @brief Extra key-value pair setting for this field
     *
     * Currently vector field need to input "dim":"x" to specify dimension.
     */
    std::map<std::string, std::string> type_params_;
};
}  // namespace milvus
