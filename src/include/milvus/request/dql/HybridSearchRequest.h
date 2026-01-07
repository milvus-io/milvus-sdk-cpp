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

#include "../../types/Function.h"
#include "../../types/SubSearchRequest.h"
#include "./DQLRequestBase.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::HybridSearch()
 */
class HybridSearchRequest : public DQLRequestBase<HybridSearchRequest> {
 public:
    /**
     * @brief Constructor
     */
    HybridSearchRequest() = default;

    /**
     * @brief Get sub search requests.
     */
    const std::vector<SubSearchRequestPtr>&
    SubRequests() const;

    /**
     * @brief Set sub search requests.
     */
    void
    SetSubRequests(std::vector<SubSearchRequestPtr>&& requests);

    /**
     * @brief Set sub search requests.
     */
    HybridSearchRequest&
    WithSubRequests(std::vector<SubSearchRequestPtr>&& requests);

    /**
     * @brief Add sub search request.
     */
    HybridSearchRequest&
    AddSubRequest(const SubSearchRequestPtr& request);

    /**
     * @brief Get rerank
     */
    FunctionPtr
    Rerank() const;

    /**
     * @brief Set rerank, such as RRF/Weighted function.
     * Read the doc for more info: https://milvus.io/docs/reranking.md
     */
    Status
    SetRerank(const FunctionPtr& rerank);

    /**
     * @brief Set rerank, suc as RRF/Weighted function.
     * Read the doc for more info: https://milvus.io/docs/reranking.md
     */
    HybridSearchRequest&
    WithRerank(const FunctionPtr& rerank);

    /**
     * @brief Get search limit(topk)
     */
    int64_t
    Limit() const;

    /**
     * @brief Set search limit(topk)
     */
    Status
    SetLimit(int64_t limit);

    /**
     * @brief Set search limit(topk)
     */
    HybridSearchRequest&
    WithLimit(int64_t limit);

    /**
     * @brief Get offset value.
     */
    int64_t
    Offset() const;

    /**
     * @brief Set offset value.
     * Note: this value is stored in the ExtraParams.
     */
    void
    SetOffset(int64_t offset);

    /**
     * @brief Set offset value.
     * Note: this value is stored in the ExtraParams.
     */
    HybridSearchRequest&
    WithOffset(int64_t offset);

    /**
     * @brief Get round decimal value.
     */
    int64_t
    GetRoundDecimal() const;

    /**
     * @brief Set round decimal value.
     */
    void
    SetRoundDecimal(int64_t round_decimal);

    /**
     * @brief Set round decimal value.
     */
    HybridSearchRequest&
    WithRoundDecimal(int64_t round_decimal);

    /**
     * @brief Get ignore growing flag.
     */
    bool
    IgnoreGrowing() const;

    /**
     * @brief Set ignore growing flag.
     */
    void
    SetIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Set ignore growing flag.
     */
    HybridSearchRequest&
    WithIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Add extra parameters such as "nlist", "ef".
     */
    HybridSearchRequest&
    AddExtraParam(const std::string& key, const std::string& value);

    /**
     * @brief Get extra param
     */
    const std::unordered_map<std::string, std::string>&
    ExtraParams() const;

    /**
     * @brief Get group by field value.
     */
    std::string
    GetGroupByField() const;

    /**
     * @brief Set group by field value.
     */
    void
    SetGroupByField(const std::string& field_name);

    /**
     * @brief Set group by field value.
     */
    HybridSearchRequest&
    WithGroupByField(const std::string& field_name);

    /**
     * @brief Get group size value.
     */
    int64_t
    GroupSize() const;

    /**
     * @brief Set group size value.
     */
    void
    SetGroupSize(int64_t group_size);

    /**
     * @brief Set group size value.
     */
    HybridSearchRequest&
    WithGroupSize(int64_t group_size);

    /**
     * @brief Get strict group size flag.
     */
    bool
    StrictGroupSize() const;

    /**
     * @brief Set strict group size flag.
     */
    void
    SetStrictGroupSize(bool strict_group_size);

    /**
     * @brief Set strict group size flag.
     */
    HybridSearchRequest&
    WithStrictGroupSize(bool strict_group_size);

 private:
    std::vector<SubSearchRequestPtr> sub_requests_;
    FunctionPtr function_;

    int64_t limit_{10};
    std::unordered_map<std::string, std::string> extra_params_;
    ::milvus::ConsistencyLevel consistency_level_{ConsistencyLevel::NONE};
};

}  // namespace milvus
