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

#include <cstdint>

#include "../Status.h"
#include "FieldData.h"

namespace milvus {

/**
 * @brief A list of embeddings to search for struct field.
 * It is also used to store the target vectors for SearchRequest/SubSearchRequest.
 */
class EmbeddingList {
 public:
    /**
     * @brief Constructor
     */
    EmbeddingList() = default;

    /**
     * @brief Get target vectors.
     */
    FieldDataPtr
    TargetVectors() const;

    /**
     * @brief Get count of target vectors.
     */
    size_t
    Count() const;

    /**
     * @brief Dimension of the vectors, for embedded text, the value is 0.
     */
    int64_t
    Dim() const;

    ////////////////////////////////////////////////////////////////////////
    // single vector assigner
    /**
     * @brief Add a binary vector to search.
     */
    Status
    AddBinaryVector(const std::string& vector);

    /**
     * @brief Add a binary vector to search.
     */
    Status
    AddBinaryVector(const BinaryVecFieldData::ElementT& vector);

    /**
     * @brief Add a float vector to search.
     */
    Status
    AddFloatVector(const FloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a sparse vector to search.
     */
    Status
    AddSparseVector(const SparseFloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a sparse vector to search.
     * We support two patterns of sparse vector:
     *  1. a json dict like {"1": 0.1, "5": 0.2, "8": 0.15}.
     *  2. a json dict like {"indices": [1, 5, 8], "values": [0.1, 0.2, 0.15]}.
     */
    Status
    AddSparseVector(const nlohmann::json& vector);

    /**
     * @brief Add a float16 vector to search.
     */
    Status
    AddFloat16Vector(const Float16VecFieldData::ElementT& vector);

    /**
     * @brief Add a float16 vector to search.
     * This method automatically converts the float array to float16 binary.
     */
    Status
    AddFloat16Vector(const std::vector<float>& vector);

    /**
     * @brief Add a bfloat16 vector to search.
     */
    Status
    AddBFloat16Vector(const BFloat16VecFieldData::ElementT& vector);

    /**
     * @brief Add a bfloat16 vector to search.
     * This method automatically converts the float array to bfloat16 binary.
     */
    Status
    AddBFloat16Vector(const std::vector<float>& vector);

    /**
     * @brief Add a text to search. Only works for BM25 function.
     */
    Status
    AddEmbeddedText(const std::string& text);

    ////////////////////////////////////////////////////////////////////////
    // multi vectors assigner
    /**
     * @brief Assign binary vectors to search request.
     * This method automatically converts the string array to uint8 array.
     * Note: this method will reset the vector list.
     */
    Status
    SetBinaryVectors(const std::vector<std::string>& vectors);

    /**
     * @brief Assign binary vectors to search request.
     * Note: this method will reset the vector list.
     */
    Status
    SetBinaryVectors(std::vector<BinaryVecFieldData::ElementT>&& vectors);

    /**
     * @brief Assign float vectors to search request.
     * Note: this method will reset the vector list.
     */
    Status
    SetFloatVectors(std::vector<FloatVecFieldData::ElementT>&& vectors);

    /**
     * @brief Assign sparse vectors to search request.
     * Note: this method will reset the vector list.
     */
    Status
    SetSparseVectors(std::vector<SparseFloatVecFieldData::ElementT>&& vectors);

    /**
     * @brief Assign sparse vectors to search request.
     * Note: this method will reset the vector list.
     * We support two patterns of sparse vector:
     *  1. a json dict like {"1": 0.1, "5": 0.2, "8": 0.15}.
     *  2. a json dict like {"indices": [1, 5, 8], "values": [0.1, 0.2, 0.15]}.
     */
    Status
    SetSparseVectors(const std::vector<nlohmann::json>& vectors);

    /**
     * @brief Assign float16 vectors to search request.
     * Note: this method will reset the vector list.
     */
    Status
    SetFloat16Vectors(std::vector<Float16VecFieldData::ElementT>&& vectors);

    /**
     * @brief Assign float16 vectors to search request.
     * This method automatically converts the float array to float16 binary.
     * Note: this method will reset the vector list.
     */
    Status
    SetFloat16Vectors(const std::vector<std::vector<float>>& vectors);

    /**
     * @brief Assign bfloat16 vectors to search request.
     * Note: this method will reset the vector list.
     */
    Status
    SetBFloat16Vectors(std::vector<BFloat16VecFieldData::ElementT>&& vectors);

    /**
     * @brief Assign bfloat16 vector.
     * This method automatically converts the float array to bfloat16 binary.
     * Note: this method will reset the vector list.
     */
    Status
    SetBFloat16Vectors(const std::vector<std::vector<float>>& vectors);

    /**
     * @brief Assign texts. Only works for BM25 function.
     * Note: this method will reset the vector list.
     */
    Status
    SetEmbeddedTexts(std::vector<std::string>&& texts);

 private:
    template <typename T, typename V>
    Status
    addVector(DataType data_type, const V& vector);

    template <typename T, typename V>
    Status
    setVectors(DataType data_type, std::vector<V>&& vectors);

 private:
    FieldDataPtr target_vectors_;
    int64_t dim_{0};
};

}  // namespace milvus
