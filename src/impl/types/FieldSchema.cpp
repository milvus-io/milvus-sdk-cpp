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

#include "milvus/types/FieldSchema.h"

#include "milvus/types/Constants.h"

namespace milvus {

FieldSchema::FieldSchema() = default;

FieldSchema::FieldSchema(std::string name, DataType data_type, std::string description, bool is_primary_key,
                         bool auto_id)
    : name_(std::move(name)),
      description_(std::move(description)),
      data_type_(data_type),
      is_primary_key_(is_primary_key),
      auto_id_(auto_id) {
}

const std::string&
FieldSchema::Name() const {
    return name_;
}

void
FieldSchema::SetName(std::string name) {
    name_ = std::move(name);
}

const std::string&
FieldSchema::Description() const {
    return description_;
}

void
FieldSchema::SetDescription(std::string description) {
    description_ = std::move(description);
}

DataType
FieldSchema::FieldDataType() const {
    return data_type_;
}

void
FieldSchema::SetDataType(DataType dt) {
    data_type_ = dt;
}

bool
FieldSchema::IsPrimaryKey() const {
    return is_primary_key_;
}

void
FieldSchema::SetPrimaryKey(bool is_primary_key) {
    is_primary_key_ = is_primary_key;
}

bool
FieldSchema::AutoID() const {
    return auto_id_;
}

void
FieldSchema::SetAutoID(bool auto_id) {
    auto_id_ = auto_id;
}

const std::map<std::string, std::string>&
FieldSchema::TypeParams() const {
    return type_params_;
}

void
FieldSchema::SetTypeParams(const std::map<std::string, std::string>& params) {
    type_params_ = params;
}

void
FieldSchema::SetTypeParams(std::map<std::string, std::string>&& params) {
    type_params_ = std::move(params);
}

uint32_t
FieldSchema::Dimension() const {
    auto iter = type_params_.find(FieldDim());
    if (iter != type_params_.end()) {
        return std::strtol(iter->second.c_str(), nullptr, 10);
    }
    return 0;
}

bool
FieldSchema::SetDimension(uint32_t dimension) {
    if (dimension == 0) {
        return false;
    }
    type_params_[FieldDim()] = std::to_string(dimension);
    return true;
}

FieldSchema&
FieldSchema::WithDimension(uint32_t dimension) {
    (void)SetDimension(dimension);
    return *this;
}

uint32_t
FieldSchema::MaxLength() const {
    auto iter = type_params_.find(FieldMaxLength());
    if (iter != type_params_.end()) {
        return std::strtol(iter->second.c_str(), nullptr, 10);
    }
    return 0;
}

void
FieldSchema::SetMaxLength(uint32_t length) {
    type_params_[FieldMaxLength()] = std::to_string(length);
}

FieldSchema&
FieldSchema::WithMaxLength(uint32_t length) {
    SetMaxLength(length);
    return *this;
}

}  // namespace milvus
