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
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Function class for hybrid search rerank and future BM25/TEXTEMBEDDING usages
 */
class MILVUS_SDK_API Function {
 public:
    /**
     * @brief Constructor
     */
    Function();

    /**
     * @brief Destructor
     */
    virtual ~Function();

    /**
     * @brief Constructor
     * @param [in] name the name.
     * @param [in] function_type the function type.
     * @param [in] description the description.
     */
    Function(std::string name, FunctionType function_type, std::string description = "");

    /**
     * @brief Name of this function, cannot be empty.
     * @return the name.
     */
    const std::string&
    Name() const;

    /**
     * @brief Set name of the function.
     * @param [in] name the name.
     */
    Status
    SetName(std::string name);

    /**
     * @brief Description of this function, can be empty.
     * @return the description.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set description of the function.
     * @param [in] description the description.
     */
    Status
    SetDescription(std::string description);

    /**
     * @brief Function type.
     * @return the function type.
     */
    FunctionType
    GetFunctionType() const;

    /**
     * @brief Set function type.
     * @param [in] function_type the function type.
     */
    virtual Status
    SetFunctionType(FunctionType function_type);

    /**
     * @brief Get input field names.
     * @return the input field names.
     */
    const std::vector<std::string>&
    InputFieldNames() const;

    /**
     * @brief Add input field name.
     * @param [in] name the name.
     */
    Status
    AddInputFieldName(std::string name);

    /**
     * @brief Get output field names.
     * @return the output field names.
     */
    const std::vector<std::string>&
    OutputFieldNames() const;

    /**
     * @brief Add output field name.
     * @param [in] name the name.
     */
    Status
    AddOutputFieldName(std::string name);

    /**
     * @brief Add extra param.
     * @param [in] key the key.
     * @param [in] value the value.
     */
    virtual Status
    AddParam(const std::string& key, const std::string& value);

    /**
     * @brief Get extra param.
     * @return the params.
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
class MILVUS_SDK_API RRFRerank : public Function {
 public:
    /**
     * @brief Constructor
     */
    RRFRerank();

    /**
     * @brief Construct an RRF rerank function with the given rank depth.
     *
     * @param [in] k rank depth.
     */
    explicit RRFRerank(int k);

    /**
     * @brief Override this method, only allow to set RERANK function type.
     * @param [in] function_type the function type.
     */
    Status
    SetFunctionType(FunctionType function_type) override;

    /**
     * @brief Set K value.
     * @param [in] k the k.
     */
    Status
    SetK(int k);
};

////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Weighted rerank function
 */
class MILVUS_SDK_API WeightedRerank : public Function {
 public:
    /**
     * @brief Construct a weighted rerank function with the given weights.
     *
     * @param [in] weights per-query weights.
     */
    explicit WeightedRerank(const std::vector<float>& weights);

    /**
     * @brief Override this method, only allow to set RERANK function type.
     * @param [in] function_type the function type.
     */
    Status
    SetFunctionType(FunctionType function_type) override;

    /**
     * @brief Set weighted values.
     * @param [in] weights the weights.
     */
    Status
    SetWeights(const std::vector<float>& weights);
};

////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Boost rerank function
 */
class MILVUS_SDK_API BoostRerank : public Function {
 public:
    /**
     * @brief Construct a boost rerank function.
     *
     * @param [in] name function name.
     */
    explicit BoostRerank(std::string name);

    /**
     * @brief Override this method, only allow to set RERANK function type.
     * @param [in] function_type the function type.
     */
    Status
    SetFunctionType(FunctionType function_type) override;

    /**
     * @brief Set filter.
     * @param [in] filter the filter.
     */
    void
    SetFilter(const std::string& filter);

    /**
     * @brief Set filter.
     * @param [in] weight the weight.
     */
    void
    SetWeight(float weight);

    /**
     * @brief Set field to do random score.
     * @param [in] field the field.
     */
    void
    SetRandomScoreField(const std::string& field);

    /**
     * @brief Set random score seed.
     * @param [in] seed the seed.
     */
    void
    SetRandomScoreSeed(int64_t seed);
};

////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Decay rerank function
 */
class MILVUS_SDK_API DecayRerank : public Function {
 public:
    /**
     * @brief Construct a decay rerank function.
     *
     * @param [in] name function name.
     */
    explicit DecayRerank(std::string name);

    /**
     * @brief Override this method, only allow to set RERANK function type.
     * @param [in] function_type the function type.
     */
    Status
    SetFunctionType(FunctionType function_type) override;

    /**
     * @brief Set decay function. "gauss", "exp", or "linear".
     * @param [in] name the name.
     */
    void
    SetFunction(const std::string& name);

    /**
     * @brief Set the reference point from which decay score is calculated.
     * Decay function can be applied on INT8/INT16/INT32/INT64/FLOAT/DOUBLE fields,
     * the origin value can be these types.
     * @param [in] val the val.
     */
    template <typename T>
    void
    SetOrigin(T val) {
        AddParam("origin", std::to_string(val));
    }

    /**
     * @brief Set a "no-decay zone" around the origin where items maintain full scores.
     * Decay function can be applied on INT8/INT16/INT32/INT64/FLOAT/DOUBLE fields,
     * the offset value can be these types.
     * @param [in] val the val.
     */
    template <typename T>
    void
    SetOffset(T val) {
        AddParam("offset", std::to_string(val));
    }

    /**
     * @brief Set position at which relevance drops to the decay value.
     * Decay function can be applied on INT8/INT16/INT32/INT64/FLOAT/DOUBLE fields,
     * the scale value can be these types.
     * @param [in] val the val.
     */
    template <typename T>
    void
    SetScale(T val) {
        AddParam("scale", std::to_string(val));
    }

    /**
     * @brief Set score value at the "scale" position,
     * @param [in] val the val.
     */
    void
    SetDecay(float val);
};

////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Model rerank function
 */
class MILVUS_SDK_API ModelRerank : public Function {
 public:
    /**
     * @brief Construct a model-based rerank function.
     *
     * @param [in] name function name.
     */
    explicit ModelRerank(std::string name);

    /**
     * @brief Override this method, only allow to set RERANK function type.
     * @param [in] function_type the function type.
     */
    Status
    SetFunctionType(FunctionType function_type) override;

    /**
     * @brief Set the model service provider to use for reranking.
     * @param [in] name the name.
     */
    void
    SetProvider(const std::string& name);

    /**
     * @brief Set the list of query strings used by the reranking model to calculate relevance scores.
     * Note: The number of query strings must match exactly the number of queries in your search operation,
     * even when using query vectors instead of text.
     * @param [in] queries the queries.
     */
    void
    SetQueries(const std::vector<std::string>& queries);

    /**
     * @brief Set the URL of the model service.
     * @param [in] url the URL.
     */
    void
    SetEndpoint(const std::string& url);

    /**
     * @brief Set the maximum number of documents to process in a single batch.
     * @param [in] val the val.
     */
    void
    SetMaxClientBatchSize(int64_t val);
};

}  // namespace milvus
