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
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Sub request for HybridSearchArguments for MilvusClient::HybridSearch().
 */
class MILVUS_SDK_API SubSearchRequest : public SearchRequestVectorAssigner<SubSearchRequest> {
 public:
    /**
     * @brief Specifies the metric type.
     * @param [in] metric_type the metric type.
     */
    SubSearchRequest&
    WithMetricType(::milvus::MetricType metric_type);

    /**
     * @brief Set search limit(topk).
     * Note: this value is stored in the ExtraParams.
     * @param [in] limit the limit.
     */
    SubSearchRequest&
    WithLimit(int64_t limit);

    /**
     * @brief Set filter expression.
     * @param [in] filter the filter.
     */
    SubSearchRequest&
    WithFilter(std::string filter);

    /**
     * @brief Set target field of ann search.
     * @param [in] ann_field the ANN field.
     */
    SubSearchRequest&
    WithAnnsField(const std::string& ann_field);

    /**
     * @brief Set timezone, takes effect for Timestamptz field.
     * @param [in] timezone the timezone.
     */
    SubSearchRequest&
    WithTimezone(const std::string& timezone);
};

using SubSearchRequestPtr = std::shared_ptr<SubSearchRequest>;

}  // namespace milvus
