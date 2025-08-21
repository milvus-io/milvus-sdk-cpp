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
#include <memory>
#include <string>
#include <unordered_map>

#include "../Status.h"
#include "IndexState.h"
#include "IndexType.h"
#include "MetricType.h"

namespace milvus {

/**
 * @brief Index description. Used by MilvusClient::CreateIndex() and MilvusClient::DescribeIndex().
 */
class IndexDesc {
 public:
    /**
     * @brief Construct a new Index Desc object
     */
    IndexDesc();

    /**
     * @brief Construct a new Index Desc object
     *
     * @param field_name field name which the index belong to
     * @param index_name index name
     * @param index_type  index type see IndexType
     * @param metric_type  metric type see MetricType, no need to set this for scalar field index
     */
    IndexDesc(std::string field_name, std::string index_name, milvus::IndexType index_type,
              milvus::MetricType metric_type = milvus::MetricType::INVALID);

    /**
     * @brief Filed name which the index belong to.
     */
    const std::string&
    FieldName() const;

    /**
     * @brief Set field name which the index belong to.
     */
    Status
    SetFieldName(std::string field_name);

    /**
     * @brief Index name. Index name cannot be empty.
     */
    const std::string&
    IndexName() const;

    /**
     * @brief Set index name.
     */
    Status
    SetIndexName(std::string index_name);

    /**
     * @brief Index ID.
     */
    int64_t
    IndexId() const;

    /**
     * @brief Set index id.
     */
    Status
    SetIndexId(int64_t index_id);

    /**
     * @brief Metric type.
     */
    milvus::MetricType
    MetricType() const;

    /**
     * @brief Set metric type.
     */
    Status
    SetMetricType(milvus::MetricType metric_type);

    /**
     * @brief Index type.
     */
    milvus::IndexType
    IndexType() const;

    /**
     * @brief Set index type.
     */
    Status
    SetIndexType(milvus::IndexType index_type);

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
     * @brief Construct a new Index Desc:: From Json object
     * @param json Json string for parse
     */
    Status
    ExtraParamsFromJson(std::string json);

    Status
    SetStateCode(const milvus::IndexStateCode& code);

    milvus::IndexStateCode
    StateCode() const;

    Status
    SetFailReason(const std::string& reason);

    std::string
    FailReason() const;

    Status
    SetIndexedRows(int64_t rows);

    int64_t
    IndexedRows() const;

    Status
    SetTotalRows(int64_t rows);

    int64_t
    TotalRows() const;

    Status
    SetPendingRows(int64_t rows);

    int64_t
    PendingRows() const;

 private:
    std::string field_name_;
    std::string index_name_;
    milvus::MetricType metric_type_{milvus::MetricType::INVALID};
    milvus::IndexType index_type_{milvus::IndexType::INVALID};
    std::unordered_map<std::string, std::string> extra_params_;

    // the following members are only for DescribeIndex
    int64_t index_id_{0};
    IndexStateCode state_code_{IndexStateCode::NONE};
    std::string failed_reason_;
    int64_t indexed_rows_{0};
    int64_t total_rows_{0};
    int64_t pending_rows_{0};
};

}  // namespace milvus
