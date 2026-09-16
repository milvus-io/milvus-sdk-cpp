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
#include <milvus/thirdparty/nlohmann/json.hpp>

#include "Function.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Function container class for search rerank.
 */
class MILVUS_SDK_API FunctionScore {
 public:
    /**
     * @brief Constructor
     */
    FunctionScore() = default;

    /**
     * @brief Get fuctions.
     * @return the functions.
     */
    const std::vector<FunctionPtr>&
    Functions() const;

    /**
     * @brief Set fuctions.
     * For Search(), the functions can be Boost/Decay/Model, etc.
     * For HybridSearch(), the functions can be RRF/Weighted, etc
     * @param [in] functions the functions.
     */
    void
    SetFunctions(std::vector<FunctionPtr>&& functions);

    /**
     * @brief Set fuctions.
     * For Search(), the functions can be Boost/Decay/Model, etc.
     * For HybridSearch(), the functions can be RRF/Weighted, etc
     * @param [in] functions the functions.
     */
    FunctionScore&
    WithFunctions(std::vector<FunctionPtr>&& functions);

    /**
     * @brief Add a fuction.
     * For Search(), the functions can be Boost/Decay/Model, etc.
     * For HybridSearch(), the functions can be RRF/Weighted, etc
     */
    FunctionScore&
    AddFunction(const FunctionPtr& function);

    /**
     * @brief Get extra params.
     * @return the params.
     */
    const std::unordered_map<std::string, nlohmann::json>&
    Params() const;

    /**
     * @brief Set extra params.
     * @param [in] params the params.
     */
    void
    SetParams(std::unordered_map<std::string, nlohmann::json>&& params);

    /**
     * @brief Set extra params.
     * @param [in] params the params.
     */
    FunctionScore&
    WithParams(std::unordered_map<std::string, nlohmann::json>&& params);

    /**
     * @brief Add an extra param.
     * @param [in] key the key.
     * @param [in] param the param.
     */
    FunctionScore&
    AddParam(const std::string& key, nlohmann::json&& param);

 protected:
    std::vector<FunctionPtr> functions_;
    std::unordered_map<std::string, nlohmann::json> params_;
};

using FunctionScorePtr = std::shared_ptr<FunctionScore>;

}  // namespace milvus
