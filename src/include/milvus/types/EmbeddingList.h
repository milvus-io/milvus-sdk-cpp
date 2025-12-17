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
 */
class EmbeddingList {
 public:
    /**
     * @brief Constructor
     */
    EmbeddingList() = default;

    /**
     * @brief Get target vectors
     */
    FieldDataPtr
    TargetVectors() const;

    /**
     * @brief Get count of target vectors
     */
    size_t
    Count() const;

    /**
     * @brief Add a binary vector to search
     */
    Status
    AddBinaryVector(const std::string& vector);

    /**
     * @brief Add a binary vector to search
     */
    Status
    AddBinaryVector(const BinaryVecFieldData::ElementT& vector);

    /**
     * @brief Add a float vector to search
     */
    Status
    AddFloatVector(const FloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a sparse vector to search
     */
    Status
    AddSparseVector(const SparseFloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a sparse vector to search. \n
     * We support two patterns of sparse vector: \n
     *  1. a json dict like {"1": 0.1, "5": 0.2, "8": 0.15}
     *  2. a json dict like {"indices": [1, 5, 8], "values": [0.1, 0.2, 0.15]}
     */
    Status
    AddSparseVector(const nlohmann::json& vector);

    /**
     * @brief Add a float16 vector to search.
     */
    Status
    AddFloat16Vector(const Float16VecFieldData::ElementT& vector);

    /**
     * @brief Add a float16 vector to search. \n
     * This method automatically converts the float array to float16 binary
     */
    Status
    AddFloat16Vector(const std::vector<float>& vector);

    /**
     * @brief Add a bfloat16 vector to search.
     */
    Status
    AddBFloat16Vector(const BFloat16VecFieldData::ElementT& vector);

    /**
     * @brief Add a bfloat16 vector to search. \n
     * This method automatically converts the float array to bfloat16 binary
     */
    Status
    AddBFloat16Vector(const std::vector<float>& vector);

    /**
     * @brief Add a text to search. Only works for BM25 function \n
     */
    Status
    AddEmbeddedText(const std::string& text);

    /**
     * @brief Dimension of the vectors, for embedded text, the value is 0
     */
    int64_t
    Dim() const;

 private:
    template <typename T, typename V>
    Status
    addVector(DataType data_type, const V& vector);

 private:
    FieldDataPtr target_vectors_;
    int64_t dim_{0};
};

}  // namespace milvus
