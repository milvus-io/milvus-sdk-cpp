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

#include "milvus/types/Function.h"

#include <nlohmann/json.hpp>

#include "../utils/Constants.h"

namespace milvus {

Function::Function() = default;

Function::Function(std::string name, FunctionType function_type, std::string description)
    : name_(std::move(name)), description_(std::move(description)), function_type_(function_type) {
}

Function::~Function() {
}

const std::string&
Function::Name() const {
    return name_;
}

Status
Function::SetName(std::string name) {
    if (name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Function name cannot be empty!"};
    }
    name_ = std::move(name);
    return Status::OK();
}

const std::string&
Function::Description() const {
    return description_;
}

Status
Function::SetDescription(std::string description) {
    description_ = std::move(description);
    return Status::OK();
}

FunctionType
Function::GetFunctionType() const {
    return function_type_;
}

Status
Function::SetFunctionType(FunctionType function_type) {
    function_type_ = function_type;
    return Status::OK();
}

const std::vector<std::string>&
Function::InputFieldNames() const {
    return input_field_names_;
}

Status
Function::AddInputFieldName(std::string name) {
    if (name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
    }
    input_field_names_.emplace_back(std::move(name));

    return Status::OK();
}

const std::vector<std::string>&
Function::OutputFieldNames() const {
    return output_field_names_;
}

Status
Function::AddOutputFieldName(std::string name) {
    if (name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
    }
    output_field_names_.emplace_back(std::move(name));
    return Status::OK();
}

Status
Function::AddParam(const std::string& key, const std::string& value) {
    params_[key] = value;
    return Status::OK();
}

const std::unordered_map<std::string, std::string>&
Function::Params() const {
    return params_;
}

////////////////////////////////////////////////////////////////////////////////////////////
// RRFRerank
RRFRerank::RRFRerank() {
    function_type_ = FunctionType::RERANK;
    params_[KeyStrategy()] = "rrf";
    SetK(60);
}

RRFRerank::RRFRerank(int k) {
    function_type_ = FunctionType::RERANK;
    params_[KeyStrategy()] = "rrf";
    SetK(k);
}

Status
RRFRerank::SetFunctionType(FunctionType function_type) {
    if (function_type != FunctionType::RERANK) {
        return {StatusCode::INVALID_AGUMENT, "RRFRerank only accepts RERANK type!"};
    }
    return Function::SetFunctionType(function_type);
}

Status
RRFRerank::SetK(int k) {
    nlohmann::json json_params;
    json_params["k"] = k;

    AddParam(KeyParams(), json_params.dump());
    return Status::OK();
}

////////////////////////////////////////////////////////////////////////////////////////////
// WeightedRerank
WeightedRerank::WeightedRerank(const std::vector<float>& weights) {
    function_type_ = FunctionType::RERANK;
    params_[KeyStrategy()] = "weighted";
    SetWeights(weights);
}

Status
WeightedRerank::SetFunctionType(FunctionType function_type) {
    if (function_type != FunctionType::RERANK) {
        return {StatusCode::INVALID_AGUMENT, "WeightedRerank only accepts RERANK type!"};
    }
    return Function::SetFunctionType(function_type);
}

Status
WeightedRerank::SetWeights(const std::vector<float>& weights) {
    nlohmann::json json_params;
    json_params["weights"] = weights;

    AddParam(KeyParams(), json_params.dump());
    return Status::OK();
}

}  // namespace milvus
