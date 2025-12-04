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

#include "milvus/types/FunctionScore.h"

namespace milvus {

const std::vector<FunctionPtr>&
FunctionScore::Functions() const {
    return functions_;
}

void
FunctionScore::SetFunctions(std::vector<FunctionPtr>&& functions) {
    functions_ = std::move(functions);
}

FunctionScore&
FunctionScore::WithFunctions(std::vector<FunctionPtr>&& functions) {
    SetFunctions(std::move(functions));
    return *this;
}

FunctionScore&
FunctionScore::AddFunction(const FunctionPtr& function) {
    functions_.push_back(function);
    return *this;
}

const std::unordered_map<std::string, nlohmann::json>&
FunctionScore::Params() const {
    return params_;
}

void
FunctionScore::SetParams(std::unordered_map<std::string, nlohmann::json>&& params) {
    params_ = std::move(params);
}

FunctionScore&
FunctionScore::WithParams(std::unordered_map<std::string, nlohmann::json>&& params) {
    SetParams(std::move(params));
    return *this;
}

FunctionScore&
FunctionScore::AddParam(const std::string& key, nlohmann::json&& param) {
    params_[key] = std::move(param);
    return *this;
}

}  // namespace milvus
