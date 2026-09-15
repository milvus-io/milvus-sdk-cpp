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
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::HybridSearch()
 * @par Example
 * @code
 * milvus::HybridSearchResponse response;
 * milvus::HybridSearchRequest request;
 * request.WithCollectionName("demo").WithLimit(3);
 * auto sub = std::make_shared<milvus::SubSearchRequest>();
 * sub->WithAnnsField("vector").WithLimit(3);
 * sub->AddFloatVector(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f});
 * request.AddSubRequest(sub).WithRerank(std::make_shared<milvus::RRFRerank>(60));
 * auto status = client->HybridSearch(request, response);
 * @endcode
 */
class MILVUS_SDK_API HybridSearchRequest : public DQLRequestBase<HybridSearchRequest> {
 public:
    /**
     * @brief Constructor
     */
    HybridSearchRequest() = default;

    /**
     * @brief Get sub search requests.
     * @return the sub requests.
     */
    const std::vector<SubSearchRequestPtr>&
    SubRequests() const;

    /**
     * @brief Set sub search requests.
     * @param [in] requests the requests.
     */
    void
    SetSubRequests(std::vector<SubSearchRequestPtr>&& requests);

    /**
     * @brief Set sub search requests.
     * @param [in] requests the requests.
     */
    HybridSearchRequest&
    WithSubRequests(std::vector<SubSearchRequestPtr>&& requests);

    /**
     * @brief Add sub search request.
     * @param [in] request the request.
     */
    HybridSearchRequest&
    AddSubRequest(const SubSearchRequestPtr& request);

    /**
     * @brief Get rerank
     * @return the rerank.
     */
    FunctionPtr
    Rerank() const;

    /**
     * @brief Set rerank, such as RRF/Weighted function.
     * Read the doc for more info: https://milvus.io/docs/reranking.md
     * @param [in] rerank the rerank.
     */
    Status
    SetRerank(const FunctionPtr& rerank);

    /**
     * @brief Set rerank, suc as RRF/Weighted function.
     * Read the doc for more info: https://milvus.io/docs/reranking.md
     * @param [in] rerank the rerank.
     */
    HybridSearchRequest&
    WithRerank(const FunctionPtr& rerank);

    /**
     * @brief Get search limit(topk)
     * @return the limit.
     */
    int64_t
    Limit() const;

    /**
     * @brief Set search limit(topk)
     * @param [in] limit the limit.
     */
    Status
    SetLimit(int64_t limit);

    /**
     * @brief Set search limit(topk)
     * @param [in] limit the limit.
     */
    HybridSearchRequest&
    WithLimit(int64_t limit);

    /**
     * @brief Get offset value.
     * @return the offset.
     */
    int64_t
    Offset() const;

    /**
     * @brief Set offset value.
     * Note: this value is stored in the ExtraParams.
     * @param [in] offset the offset.
     */
    void
    SetOffset(int64_t offset);

    /**
     * @brief Set offset value.
     * Note: this value is stored in the ExtraParams.
     * @param [in] offset the offset.
     */
    HybridSearchRequest&
    WithOffset(int64_t offset);

    /**
     * @brief Get round decimal value.
     * @return the round decimal.
     */
    int64_t
    GetRoundDecimal() const;

    /**
     * @brief Set round decimal value.
     * @param [in] round_decimal the round decimal.
     */
    void
    SetRoundDecimal(int64_t round_decimal);

    /**
     * @brief Set round decimal value.
     * @param [in] round_decimal the round decimal.
     */
    HybridSearchRequest&
    WithRoundDecimal(int64_t round_decimal);

    /**
     * @brief Get ignore growing flag.
     * @return the ignore growing.
     */
    bool
    IgnoreGrowing() const;

    /**
     * @brief Set ignore growing flag.
     * @param [in] ignore_growing the ignore growing.
     */
    void
    SetIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Set ignore growing flag.
     * @param [in] ignore_growing the ignore growing.
     */
    HybridSearchRequest&
    WithIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Add extra parameters such as "nlist", "ef".
     * @param [in] key the key.
     * @param [in] value the value.
     */
    HybridSearchRequest&
    AddExtraParam(const std::string& key, const std::string& value);

    /**
     * @brief Get extra param
     * @return the extra params.
     */
    const std::unordered_map<std::string, std::string>&
    ExtraParams() const;

    /**
     * @brief Get group by field value.
     * @return the group by field name.
     */
    std::string
    GetGroupByField() const;

    /**
     * @brief Set group by field value.
     * @param [in] field_name the field name.
     */
    void
    SetGroupByField(const std::string& field_name);

    /**
     * @brief Set group by field value.
     * @param [in] field_name the field name.
     */
    HybridSearchRequest&
    WithGroupByField(const std::string& field_name);

    /**
     * @brief Get group size value.
     * @return the group size.
     */
    int64_t
    GroupSize() const;

    /**
     * @brief Set group size value.
     * @param [in] group_size the group size.
     */
    void
    SetGroupSize(int64_t group_size);

    /**
     * @brief Set group size value.
     * @param [in] group_size the group size.
     */
    HybridSearchRequest&
    WithGroupSize(int64_t group_size);

    /**
     * @brief Get strict group size flag.
     * @return the strict group size.
     */
    bool
    StrictGroupSize() const;

    /**
     * @brief Set strict group size flag.
     * @param [in] strict_group_size the strict group size.
     */
    void
    SetStrictGroupSize(bool strict_group_size);

    /**
     * @brief Set strict group size flag.
     * @param [in] strict_group_size the strict group size.
     */
    HybridSearchRequest&
    WithStrictGroupSize(bool strict_group_size);

    /**
     * @brief Validate the hybrid search request before sending it to the server.
     * @return Status::OK when the request is valid, otherwise an INVALID_ARGUMENT error.
     */
    Status
    Validate() const;

 private:
    std::vector<SubSearchRequestPtr> sub_requests_;
    FunctionPtr function_;

    int64_t limit_{10};
    std::unordered_map<std::string, std::string> extra_params_;
    ::milvus::ConsistencyLevel consistency_level_{ConsistencyLevel::NONE};
};

}  // namespace milvus
