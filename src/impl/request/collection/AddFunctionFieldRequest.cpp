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

#include "milvus/request/collection/AddFunctionFieldRequest.h"

#include <utility>

namespace milvus {

const FieldSchema&
AddFunctionFieldRequest::Field() const {
    return field_;
}

void
AddFunctionFieldRequest::SetField(FieldSchema&& field_schema) {
    field_ = std::move(field_schema);
}

AddFunctionFieldRequest&
AddFunctionFieldRequest::WithField(FieldSchema&& field_schema) {
    SetField(std::move(field_schema));
    return *this;
}

const FunctionPtr&
AddFunctionFieldRequest::Function() const {
    return function_;
}

void
AddFunctionFieldRequest::SetFunction(const FunctionPtr& function) {
    function_ = function;
}

AddFunctionFieldRequest&
AddFunctionFieldRequest::WithFunction(const FunctionPtr& function) {
    SetFunction(function);
    return *this;
}

const IndexDesc&
AddFunctionFieldRequest::Index() const {
    return index_;
}

void
AddFunctionFieldRequest::SetIndex(IndexDesc&& index) {
    index_ = std::move(index);
}

AddFunctionFieldRequest&
AddFunctionFieldRequest::WithIndex(IndexDesc&& index) {
    SetIndex(std::move(index));
    return *this;
}

}  // namespace milvus
