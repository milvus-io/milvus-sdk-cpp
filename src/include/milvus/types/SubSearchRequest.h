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
#include <set>
#include <string>
#include <unordered_map>

#include "../Status.h"
#include "Constants.h"
#include "FieldData.h"
#include "MetricType.h"

namespace milvus {

/**
 * @brief A base class for SubSearchRequest and SearchArguments
 */
class SearchRequestBase {
 protected:
    SearchRequestBase() = default;
    virtual ~SearchRequestBase() = default;

 public:
    /**
     * @brief Get filter expression
     */
    const std::string&
    Filter() const;

    /**
     * @brief Set filter expression
     */
    Status
    SetFilter(std::string filter);

    /**
     * @brief Get target vectors
     */
    FieldDataPtr
    TargetVectors() const;

    /**
     * @brief Add a binary vector to search
     */
    Status
    AddBinaryVector(std::string field_name, const std::string& vector);

    /**
     * @brief Add a binary vector to search
     */
    Status
    AddBinaryVector(std::string field_name, const BinaryVecFieldData::ElementT& vector);

    /**
     * @brief Add a float vector to search
     */
    Status
    AddFloatVector(std::string field_name, const FloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a sparse vector to search
     */
    Status
    AddSparseVector(std::string field_name, const SparseFloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a sparse vector to search. \n
     * We support two patterns of sparse vector: \n
     *  1. a json dict like {"1": 0.1, "5": 0.2, "8": 0.15}
     *  2. a json dict like {"indices": [1, 5, 8], "values": [0.1, 0.2, 0.15]}
     */
    Status
    AddSparseVector(std::string field_name, const nlohmann::json& vector);

    /**
     * @brief Add a float16 vector to search.
     */
    Status
    AddFloat16Vector(std::string field_name, const Float16VecFieldData::ElementT& vector);

    /**
     * @brief Add a float16 vector to search. \n
     * This method automatically converts the float array to float16 binary
     */
    Status
    AddFloat16Vector(std::string field_name, const std::vector<float>& vector);

    /**
     * @brief Add a bfloat16 vector to search.
     */
    Status
    AddBFloat16Vector(std::string field_name, const BFloat16VecFieldData::ElementT& vector);

    /**
     * @brief Add a bfloat16 vector to search. \n
     * This method automatically converts the float array to bfloat16 binary
     */
    Status
    AddBFloat16Vector(std::string field_name, const std::vector<float>& vector);

    /**
     * @brief Add a text to search. Only works for BM25 function \n
     */
    Status
    AddEmbeddedText(std::string field_name, const std::string& text);

    /**
     * @brief Get anns field name
     */
    std::string
    AnnsField() const;

    /**
     * @brief Get search limit(topk)
     */
    int64_t
    Limit() const;

    /**
     * @brief Set search limit(topk)
     * Note: this value is stored in the ExtraParams
     */
    Status
    SetLimit(int64_t limit);

    /**
     * @brief Get the metric type
     */
    ::milvus::MetricType
    MetricType() const;

    /**
     * @brief Specifies the metric type
     */
    Status
    SetMetricType(::milvus::MetricType metric_type);

    /**
     * @brief Add extra param
     * Note: int v2.4, we redefine this method, old client code might be affected
     */
    Status
    AddExtraParam(const std::string& key, const std::string& value);

    /**
     * @brief Get extra param
     * Note: int v2.4, we redefine this method, old client code might be affected
     */
    const std::unordered_map<std::string, std::string>&
    ExtraParams() const;

    /**
     * @brief Get range radius
     * @return
     */
    double
    Radius() const;

    /**
     * @brief Set range radius
     * Note: this value is stored in the ExtraParams
     * @return
     */
    Status
    SetRadius(double value);

    /**
     * @brief Get range filter
     * @return
     */
    double
    RangeFilter() const;

    /**
     * @brief Set range filter
     * Note: this value is stored in the ExtraParams
     * @return
     */
    Status
    SetRangeFilter(double value);

    /**
     * @brief Set range radius
     * @param range_filter while radius sets the outer limit of the search, range_filter can be optionally used to
     * define an inner boundary, creating a distance range within which vectors must fall to be considered matches.
     * @param radius defines the outer boundary of your search space. Only vectors that are within this distance from
     * the query vector are considered potential matches.
     */
    Status
    SetRange(double range_filter, double radius);

    /**
     * @brief Validate for search arguments and get name of the target anns field
     * Note: in v2.4+, a collection can have one or more vector fields. If a collection has
     * only one vector field, users can set an empty name in the AddTargetVector(),
     * the server can know the vector field name.
     * But if the collection has multiple vector fields, users need to provide a non-empty name
     * in the AddTargetVector() method, and if users call AddTargetVector() mutiple times, he must
     * ensure that the name is the same, otherwise the Validate() method will return an error.
     * The Validate() method is called before Search().
     */
    Status
    Validate() const;

 private:
    Status verifyVectorType(DataType) const;

    template <typename T, typename V>
    Status
    addVector(std::string field_name, DataType data_type, const V& vector);

 protected:
    FieldDataPtr target_vectors_;

    int64_t limit_{10};
    std::string filter_expression_;

    ::milvus::MetricType metric_type_{::milvus::MetricType::DEFAULT};

    std::unordered_map<std::string, std::string> extra_params_;
};

/**
 * @brief Sub request for HybridSearchArguments for MilvusClient::HybridSearch().
 */
class SubSearchRequest : public SearchRequestBase {};

using SubSearchRequestPtr = std::shared_ptr<SubSearchRequest>;

}  // namespace milvus
