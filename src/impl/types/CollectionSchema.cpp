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

#include "milvus/types/CollectionSchema.h"

#include <algorithm>

namespace milvus {

CollectionSchema::CollectionSchema() = default;

CollectionSchema::CollectionSchema(std::string name, std::string desc, int32_t shard_num)
    : name_(std::move(name)), description_(std::move(desc)), shard_num_(shard_num) {
}

const std::string&
CollectionSchema::Name() const {
    return name_;
}

void
CollectionSchema::SetName(std::string name) {
    name_ = std::move(name);
}

const std::string&
CollectionSchema::Description() const {
    return description_;
}

void
CollectionSchema::SetDescription(std::string description) {
    description_ = std::move(description);
}

int32_t
CollectionSchema::ShardsNum() const {
    return shard_num_;
}

void
CollectionSchema::SetShardsNum(int32_t num) {
    shard_num_ = num;
}

const std::vector<FieldSchema>&
CollectionSchema::Fields() const {
    return fields_;
}

bool
CollectionSchema::AddField(const FieldSchema& field_schema) {
    const auto it = std::find_if(fields_.cbegin(), fields_.cend(), [&field_schema](const FieldSchema& field) {
        return field.Name() == field_schema.Name();
    });
    if (it != fields_.cend()) {
        return false;
    }
    fields_.emplace_back(field_schema);
    return true;
}

bool
CollectionSchema::AddField(FieldSchema&& field_schema) {
    const auto it = std::find_if(fields_.cbegin(), fields_.cend(), [&field_schema](const FieldSchema& field) {
        return field.Name() == field_schema.Name();
    });
    if (it != fields_.cend()) {
        return false;
    }
    fields_.emplace_back(std::move(field_schema));
    return true;
}

std::unordered_set<std::string>
CollectionSchema::AnnsFieldNames() const {
    std::unordered_set<std::string> ret;
    for (const auto& field : fields_) {
        auto data_type = field.FieldDataType();
        if (data_type == DataType::BINARY_VECTOR || data_type == DataType::FLOAT_VECTOR) {
            ret.emplace(field.Name());
        }
    }
    return ret;
}

}  // namespace milvus
