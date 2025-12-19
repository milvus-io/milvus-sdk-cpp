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

#include "SearchRequestBase.h"

namespace milvus {

/**
 * @brief Sub request for HybridSearchArguments for MilvusClient::HybridSearch().
 */
class SubSearchRequest : public SearchRequestBase {
 public:
    /**
     * @brief Specifies the metric type.
     */
    SubSearchRequest&
    WithMetricType(::milvus::MetricType metric_type);

    /**
     * @brief Set search limit(topk).
     * Note: this value is stored in the ExtraParams
     */
    SubSearchRequest&
    WithLimit(int64_t limit);

    /**
     * @brief Set filter expression.
     */
    SubSearchRequest&
    WithFilter(std::string filter);

    /**
     * @brief Set target field of ann search
     */
    SubSearchRequest&
    WithAnnsField(const std::string& ann_field);

    /**
     * @brief Set timezone, takes effect for Timestamptz field.
     */
    SubSearchRequest&
    WithTimezone(const std::string& timezone);

    /**
     * @brief Add a binary vector to search
     */
    SubSearchRequest&
    AddBinaryVector(const std::string& vector);

    /**
     * @brief Add a binary vector to search
     */
    SubSearchRequest&
    AddBinaryVector(const BinaryVecFieldData::ElementT& vector);

    /**
     * @brief Add a float vector to search
     */
    SubSearchRequest&
    AddFloatVector(const FloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a sparse vector to search
     */
    SubSearchRequest&
    AddSparseVector(const SparseFloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a sparse vector to search. \n
     * We support two patterns of sparse vector: \n
     *  1. a json dict like {"1": 0.1, "5": 0.2, "8": 0.15}
     *  2. a json dict like {"indices": [1, 5, 8], "values": [0.1, 0.2, 0.15]}
     */
    SubSearchRequest&
    AddSparseVector(const nlohmann::json& vector);

    /**
     * @brief Add a float16 vector to search.
     */
    SubSearchRequest&
    AddFloat16Vector(const Float16VecFieldData::ElementT& vector);

    /**
     * @brief Add a float16 vector to search. \n
     * This method automatically converts the float array to float16 binary
     */
    SubSearchRequest&
    AddFloat16Vector(const std::vector<float>& vector);

    /**
     * @brief Add a bfloat16 vector to search.
     */
    SubSearchRequest&
    AddBFloat16Vector(const BFloat16VecFieldData::ElementT& vector);

    /**
     * @brief Add a bfloat16 vector to search. \n
     * This method automatically converts the float array to bfloat16 binary
     */
    SubSearchRequest&
    AddBFloat16Vector(const std::vector<float>& vector);

    /**
     * @brief Add a text to search. Only works for BM25 function \n
     */
    SubSearchRequest&
    AddEmbeddedText(const std::string& text);

    /**
     * @brief Add an int8 vector to search
     */
    SubSearchRequest&
    AddInt8Vector(const Int8VecFieldData::ElementT& vector);

    /**
     * @brief Add an embedding list to search on struct field
     */
    SubSearchRequest&
    AddEmbeddingList(EmbeddingList&& emb_list);
};

using SubSearchRequestPtr = std::shared_ptr<SubSearchRequest>;

}  // namespace milvus
