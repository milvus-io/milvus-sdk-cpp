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
#include "EmbeddingList.h"
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
     * @brief Get filter expression.
     */
    const std::string&
    Filter() const;

    /**
     * @brief Set filter expression.
     */
    Status
    SetFilter(std::string filter);

    /**
     * @brief Add a filter template.
     * Expression template, to improve expression parsing performance in complicated list.
     * Assume user has a filter = "pk > 3 and city in ["beijing", "shanghai", ......]
     * The long list of city will increase the time cost to parse this expression.
     * So, we provide filterTemplate for this purpose, user can set filter like this:
     *     filter = "pk > {age} and city in {city}"
     *     filterTemplate = {"age": 3, "city": ["beijing", "shanghai", ......]}
     * Valid value of a template can be:
     *     boolean, numeric, string, array.
     */
    Status
    AddFilterTemplate(std::string key, const nlohmann::json& filter_template);

    /**
     * @brief Get filter templates.
     */
    const std::unordered_map<std::string, nlohmann::json>&
    FilterTemplates() const;

    /**
     * @brief Set filter templates.
     */
    Status
    SetFilterTemplates(std::unordered_map<std::string, nlohmann::json>&& filter_templates);

    /**
     * @brief Get target vectors.
     */
    FieldDataPtr
    TargetVectors() const;

    /**
     * @brief Get embedding lists for struct field ann search.
     */
    const std::vector<EmbeddingList>&
    EmbeddingLists() const;

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

    /**
     * @brief Add an int8 vector to search.
     */
    Status
    AddInt8Vector(const Int8VecFieldData::ElementT& vector);

    /**
     * @brief Add an embedding list to search on struct field.
     */
    Status
    AddEmbeddingList(EmbeddingList&& emb_list);

    /**
     * @brief Get anns field name.
     */
    std::string
    AnnsField() const;

    /**
     * @brief Set target field of ann search.
     */
    Status
    SetAnnsField(const std::string& ann_field);

    /**
     * @brief Get search limit(topk).
     */
    int64_t
    Limit() const;

    /**
     * @brief Set search limit(topk).
     */
    Status
    SetLimit(int64_t limit);

    /**
     * @brief Get the metric type.
     */
    ::milvus::MetricType
    MetricType() const;

    /**
     * @brief Specifies the metric type.
     */
    Status
    SetMetricType(::milvus::MetricType metric_type);

    /**
     * @brief Add extra param.
     * Note: int v2.4, we redefine this method, old client code might be affected.
     */
    Status
    AddExtraParam(const std::string& key, const std::string& value);

    /**
     * @brief Get extra param.
     * Note: int v2.4, we redefine this method, old client code might be affected.
     */
    const std::unordered_map<std::string, std::string>&
    ExtraParams() const;

    /**
     * @brief Get range radius.
     * @return
     */
    double
    Radius() const;

    /**
     * @brief Set range radius.
     * Note: this value is stored in the ExtraParams.
     * @return
     */
    Status
    SetRadius(double value);

    /**
     * @brief Get range filter.
     * @return
     */
    double
    RangeFilter() const;

    /**
     * @brief Set range filter.
     * Note: this value is stored in the ExtraParams.
     * @return
     */
    Status
    SetRangeFilter(double value);

    /**
     * @brief Set range radius.
     * @param range_filter while radius sets the outer limit of the search, range_filter can be optionally used to
     * define an inner boundary, creating a distance range within which vectors must fall to be considered matches.
     * @param radius defines the outer boundary of your search space. Only vectors that are within this distance from
     * the query vector are considered potential matches.
     */
    Status
    SetRange(double range_filter, double radius);

    /**
     * @brief Get timezone, takes effect for Timestamptz field.
     * Note: this value is stored in the ExtraParams.
     */
    std::string
    Timezone() const;

    /**
     * @brief Set timezone, takes effect for Timestamptz field.
     * Note: this value is stored in the ExtraParams.
     */
    Status
    SetTimezone(const std::string& timezone);

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

    ///////////////////////////////////////////////////////////////////////////////////////
    // deprecated methods
    /**
     * @brief Add a binary vector to search.
     * @deprecated replaced by same name method without field_name parameter, use SetAnnField() to set ann field name.
     */
    Status
    AddBinaryVector(std::string field_name, const std::string& vector);

    /**
     * @brief Add a binary vector to search.
     * @deprecated replaced by same name method without field_name parameter
     */
    Status
    AddBinaryVector(std::string field_name, const BinaryVecFieldData::ElementT& vector);

    /**
     * @brief Add a float vector to search.
     * @deprecated replaced by same name method without field_name parameter, use SetAnnField() to set ann field name.
     */
    Status
    AddFloatVector(std::string field_name, const FloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a sparse vector to search.
     * @deprecated replaced by same name method without field_name parameter, use SetAnnField() to set ann field name.
     */
    Status
    AddSparseVector(std::string field_name, const SparseFloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a sparse vector to search.
     * We support two patterns of sparse vector:
     *  1. a json dict like {"1": 0.1, "5": 0.2, "8": 0.15}.
     *  2. a json dict like {"indices": [1, 5, 8], "values": [0.1, 0.2, 0.15]}.
     * @deprecated replaced by same name method without field_name parameter, use SetAnnField() to set ann field name.
     */
    Status
    AddSparseVector(std::string field_name, const nlohmann::json& vector);

    /**
     * @brief Add a float16 vector to search.
     * @deprecated replaced by same name method without field_name parameter, use SetAnnField() to set ann field name.
     */
    Status
    AddFloat16Vector(std::string field_name, const Float16VecFieldData::ElementT& vector);

    /**
     * @brief Add a float16 vector to search.
     * This method automatically converts the float array to float16 binary.
     * @deprecated replaced by same name method without field_name parameter, use SetAnnField() to set ann field name.
     */
    Status
    AddFloat16Vector(std::string field_name, const std::vector<float>& vector);

    /**
     * @brief Add a bfloat16 vector to search.
     * @deprecated replaced by same name method without field_name parameter, use SetAnnField() to set ann field name.
     */
    Status
    AddBFloat16Vector(std::string field_name, const BFloat16VecFieldData::ElementT& vector);

    /**
     * @brief Add a bfloat16 vector to search.
     * This method automatically converts the float array to bfloat16 binary.
     * @deprecated replaced by same name method without field_name parameter, use SetAnnField() to set ann field name.
     */
    Status
    AddBFloat16Vector(std::string field_name, const std::vector<float>& vector);

    /**
     * @brief Add a text to search. Only works for BM25 function.
     * @deprecated replaced by same name method without field_name parameter, use SetAnnField() to set ann field name.
     */
    Status
    AddEmbeddedText(std::string field_name, const std::string& text);

    /**
     * @brief Add an int8 vector to search.
     * @deprecated replaced by same name method without field_name parameter, use SetAnnField() to set ann field name.
     */
    Status
    AddInt8Vector(std::string field_name, const Int8VecFieldData::ElementT& vector);

 protected:
    std::string ann_field_;
    EmbeddingList target_vectors_;
    std::vector<EmbeddingList> embedding_lists_;  // for struct field ann search

    int64_t limit_{10};
    std::string filter_expression_;
    std::unordered_map<std::string, nlohmann::json> filter_templates_;

    ::milvus::MetricType metric_type_{::milvus::MetricType::DEFAULT};

    std::unordered_map<std::string, std::string> extra_params_;
};

}  // namespace milvus
