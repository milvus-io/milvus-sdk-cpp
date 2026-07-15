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

#include "milvus/types/OrderByField.h"

#include <utility>

namespace milvus {

OrderByField::OrderByField(std::string field_name, AggregationDirection direction)
    : field_name_(std::move(field_name)), direction_(direction) {
}

const std::string&
OrderByField::FieldName() const {
    return field_name_;
}

void
OrderByField::SetFieldName(std::string field_name) {
    field_name_ = std::move(field_name);
}

OrderByField&
OrderByField::WithFieldName(std::string field_name) {
    SetFieldName(std::move(field_name));
    return *this;
}

AggregationDirection
OrderByField::Direction() const {
    return direction_;
}

void
OrderByField::SetDirection(AggregationDirection direction) {
    direction_ = direction;
}

OrderByField&
OrderByField::WithDirection(AggregationDirection direction) {
    SetDirection(direction);
    return *this;
}

}  // namespace milvus
