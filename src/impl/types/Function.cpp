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
    params_[STRATEGY] = "rrf";
    SetK(60);
}

RRFRerank::RRFRerank(int k) {
    function_type_ = FunctionType::RERANK;
    params_[STRATEGY] = "rrf";
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

    AddParam(PARAMS, json_params.dump());
    return Status::OK();
}

////////////////////////////////////////////////////////////////////////////////////////////
// WeightedRerank
WeightedRerank::WeightedRerank(const std::vector<float>& weights) {
    function_type_ = FunctionType::RERANK;
    params_[STRATEGY] = "weighted";
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

    AddParam(PARAMS, json_params.dump());
    return Status::OK();
}

////////////////////////////////////////////////////////////////////////////////////////////
// BoostRerank
BoostRerank::BoostRerank(std::string name) : Function(name, FunctionType::RERANK) {
    params_[RERANKER] = "boost";
}

// Override SetFunctionType method
Status
BoostRerank::SetFunctionType(FunctionType function_type) {
    if (function_type != FunctionType::RERANK) {
        return {StatusCode::INVALID_AGUMENT, "BoostRerank only accepts RERANK type!"};
    }
    return Function::SetFunctionType(function_type);
}

void
BoostRerank::SetFilter(const std::string& filter) {
    if (!filter.empty()) {
        AddParam("filter", filter);
    }
}

void
BoostRerank::SetWeight(float weight) {
    if (weight > 0.0) {
        AddParam("weight", std::to_string(weight));
    }
}

void
BoostRerank::SetRandomScoreField(const std::string& field) {
    nlohmann::json temp;
    auto it = params_.find(RANDOM_SCORE);
    if (it == params_.end()) {
        temp["field"] = field;
    } else {
        temp = nlohmann::json::parse(it->second);
        temp["field"] = field;
    }
    AddParam(RANDOM_SCORE, temp.dump());
}

void
BoostRerank::SetRandomScoreSeed(int64_t seed) {
    nlohmann::json temp;
    auto it = params_.find(RANDOM_SCORE);
    if (it == params_.end()) {
        temp["seed"] = std::to_string(seed);
    } else {
        temp = nlohmann::json::parse(it->second);
        temp["seed"] = std::to_string(seed);
    }
    AddParam(RANDOM_SCORE, temp.dump());
}

////////////////////////////////////////////////////////////////////////////////////////////
// DecayRerank
DecayRerank::DecayRerank(std::string name) : Function(name, FunctionType::RERANK) {
    params_[RERANKER] = "decay";
}

// Override SetFunctionType method
Status
DecayRerank::SetFunctionType(FunctionType function_type) {
    if (function_type != FunctionType::RERANK) {
        return {StatusCode::INVALID_AGUMENT, "DecayRerank only accepts RERANK type!"};
    }
    return Function::SetFunctionType(function_type);
}

void
DecayRerank::SetFunction(const std::string& name) {
    if (!name.empty()) {
        AddParam("function", name);
    }
}

void
DecayRerank::SetDecay(float val) {
    AddParam("decay", std::to_string(val));
}

////////////////////////////////////////////////////////////////////////////////////////////
// ModelRerank
ModelRerank::ModelRerank(std::string name) : Function(name, FunctionType::RERANK) {
    params_[RERANKER] = "decay";
}

// Override SetFunctionType method
Status
ModelRerank::SetFunctionType(FunctionType function_type) {
    if (function_type != FunctionType::RERANK) {
        return {StatusCode::INVALID_AGUMENT, "ModelRerank only accepts RERANK type!"};
    }
    return Function::SetFunctionType(function_type);
}

void
ModelRerank::SetProvider(const std::string& name) {
    AddParam("provider", name);
}

void
ModelRerank::SetQueries(const std::vector<std::string>& queries) {
    nlohmann::json temp = queries;
    AddParam("queries", temp.dump());
}

void
ModelRerank::SetEndpoint(const std::string& url) {
    AddParam("endpoint", url);
}

void
ModelRerank::SetMaxClientBatchSize(int64_t val) {
    AddParam("maxBatch", std::to_string(val));
}

}  // namespace milvus
