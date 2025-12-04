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

#include <memory>
#include <nlohmann/json.hpp>

#include "Function.h"

namespace milvus {

/**
 * @brief Function container class for search rerank
 */
class FunctionScore {
 public:
    /**
     * @brief Constructor
     */
    FunctionScore() = default;

    /**
     * @brief Get fuctions
     */
    const std::vector<FunctionPtr>&
    Functions() const;

    /**
     * @brief Set fuctions
     * For Search(), the functions can be Boost/Decay/Model, etc.
     * For HybridSearch(), the functions can be RRF/Weighted, etc
     */
    void
    SetFunctions(std::vector<FunctionPtr>&& functions);

    /**
     * @brief Set fuctions
     * For Search(), the functions can be Boost/Decay/Model, etc.
     * For HybridSearch(), the functions can be RRF/Weighted, etc
     */
    FunctionScore&
    WithFunctions(std::vector<FunctionPtr>&& functions);

    /**
     * @brief Add a fuction
     * For Search(), the functions can be Boost/Decay/Model, etc.
     * For HybridSearch(), the functions can be RRF/Weighted, etc
     */
    FunctionScore&
    AddFunction(const FunctionPtr& function);

    /**
     * @brief Get extra params
     */
    const std::unordered_map<std::string, nlohmann::json>&
    Params() const;

    /**
     * @brief Set extra params
     */
    void
    SetParams(std::unordered_map<std::string, nlohmann::json>&& params);

    /**
     * @brief Set extra params
     */
    FunctionScore&
    WithParams(std::unordered_map<std::string, nlohmann::json>&& params);

    /**
     * @brief Add an extra param
     */
    FunctionScore&
    AddParam(const std::string& key, nlohmann::json&& param);

 protected:
    std::vector<FunctionPtr> functions_;
    std::unordered_map<std::string, nlohmann::json> params_;
};

using FunctionScorePtr = std::shared_ptr<FunctionScore>;

}  // namespace milvus
