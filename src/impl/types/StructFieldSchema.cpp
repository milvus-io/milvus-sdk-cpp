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

#include "milvus/types/StructFieldSchema.h"

#include <memory>

namespace milvus {

StructFieldSchema::StructFieldSchema() {
}

StructFieldSchema::StructFieldSchema(std::string name, std::string description)
    : name_(std::move(name)), description_(std::move(description)) {
}

const std::string&
StructFieldSchema::Name() const {
    return name_;
}

void
StructFieldSchema::SetName(std::string name) {
    name_ = std::move(name);
}

StructFieldSchema&
StructFieldSchema::WithName(std::string name) {
    SetName(std::move(name));
    return *this;
}

const std::string&
StructFieldSchema::Description() const {
    return description_;
}

void
StructFieldSchema::SetDescription(std::string description) {
    description_ = std::move(description);
}

StructFieldSchema&
StructFieldSchema::WithDescription(std::string description) {
    SetDescription(std::move(description));
    return *this;
}

int64_t
StructFieldSchema::MaxCapacity() const {
    return capacity_;
}

void
StructFieldSchema::SetMaxCapacity(int64_t capacity) {
    capacity_ = capacity;
}

StructFieldSchema&
StructFieldSchema::WithMaxCapacity(int64_t capacity) {
    SetMaxCapacity(capacity);
    return *this;
}

const std::vector<FieldSchema>&
StructFieldSchema::Fields() const {
    return fields_;
}

StructFieldSchema&
StructFieldSchema::AddField(const FieldSchema& field_schema) {
    fields_.push_back(field_schema);
    return *this;
}

StructFieldSchema&
StructFieldSchema::AddField(FieldSchema&& field_schema) {
    fields_.emplace_back(std::move(field_schema));
    return *this;
}

}  // namespace milvus
