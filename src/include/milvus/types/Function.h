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
#include <string>
#include <unordered_map>
#include <vector>

#include "../Status.h"
#include "FunctionType.h"

namespace milvus {

/**
 * @brief Function class for hybrid search rerank and future BM25/TEXTEMBEDDING usages
 */
class Function {
 public:
    Function();
    virtual ~Function();

    /**
     * @brief Constructor
     */
    Function(std::string name, FunctionType function_type, std::string description = "");

    /**
     * @brief Name of this function, cannot be empty.
     */
    const std::string&
    Name() const;

    /**
     * @brief Set name of the function.
     */
    Status
    SetName(std::string name);

    /**
     * @brief Description of this function, can be empty.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set description of the function.
     */
    Status
    SetDescription(std::string description);

    /**
     * @brief Function type.
     */
    FunctionType
    GetFunctionType() const;

    /**
     * @brief Set function type.
     */
    virtual Status
    SetFunctionType(FunctionType function_type);

    /**
     * @brief Get input field names
     */
    const std::vector<std::string>&
    InputFieldNames() const;

    /**
     * @brief Add input field name
     */
    Status
    AddInputFieldName(std::string name);

    /**
     * @brief Get output field names
     */
    const std::vector<std::string>&
    OutputFieldNames() const;

    /**
     * @brief Add output field name
     */
    Status
    AddOutputFieldName(std::string name);

    /**
     * @brief Add extra param
     */
    virtual Status
    AddParam(const std::string& key, const std::string& value);

    /**
     * @brief Get extra param
     */
    virtual const std::unordered_map<std::string, std::string>&
    Params() const;

 protected:
    std::string name_;
    std::string description_;
    FunctionType function_type_{FunctionType::UNKNOWN};

    std::vector<std::string> input_field_names_;
    std::vector<std::string> output_field_names_;

    std::unordered_map<std::string, std::string> params_;
};

using FunctionPtr = std::shared_ptr<Function>;

////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief RRF rerank function
 */
class RRFRerank : public Function {
 public:
    RRFRerank();
    explicit RRFRerank(int k);

    /**
     * @brief Override this method, only allow to set RERANK function type
     */
    Status
    SetFunctionType(FunctionType function_type) override;

    /**
     * @brief Set K value
     */
    Status
    SetK(int k);
};

////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Weighted rerank function
 */
class WeightedRerank : public Function {
 public:
    explicit WeightedRerank(const std::vector<float>& weights);

    /**
     * @brief Override this method, only allow to set RERANK function type
     */
    Status
    SetFunctionType(FunctionType function_type) override;

    /**
     * @brief Set weighted values
     */
    Status
    SetWeights(const std::vector<float>& weights);
};

}  // namespace milvus
