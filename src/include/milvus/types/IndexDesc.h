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
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Index description. Used by MilvusClient::CreateIndex() and MilvusClient::DescribeIndex().
 */
class MILVUS_SDK_API IndexDesc {
 public:
    /**
     * @brief Construct a new IndexDesc object.
     */
    IndexDesc();

    /**
     * @brief Construct a new IndexDesc object.
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
     * @return the field name.
     */
    const std::string&
    FieldName() const;

    /**
     * @brief Set field name which the index belong to.
     * @param [in] field_name the field name.
     */
    Status
    SetFieldName(std::string field_name);

    /**
     * @brief Index name. Index name cannot be empty.
     * @return the index name.
     */
    const std::string&
    IndexName() const;

    /**
     * @brief Set index name.
     * @param [in] index_name the index name.
     */
    Status
    SetIndexName(std::string index_name);

    /**
     * @brief Index ID.
     * @return the index ID.
     */
    int64_t
    IndexId() const;

    /**
     * @brief Set index id.
     * @param [in] index_id the index ID.
     */
    Status
    SetIndexId(int64_t index_id);

    /**
     * @brief Metric type.
     * @return the metric type.
     */
    milvus::MetricType
    MetricType() const;

    /**
     * @brief Set metric type.
     * @param [in] metric_type the metric type.
     */
    Status
    SetMetricType(milvus::MetricType metric_type);

    /**
     * @brief Index type.
     * @return the index type.
     */
    milvus::IndexType
    IndexType() const;

    /**
     * @brief Set index type.
     * @param [in] index_type the index type.
     */
    Status
    SetIndexType(milvus::IndexType index_type);

    /**
     * @brief Add extra param.
     * Note: this method was redefined in v2.4, which may affect older client code.
     * @param [in] key the key.
     * @param [in] value the value.
     */
    Status
    AddExtraParam(const std::string& key, const std::string& value);

    /**
     * @brief Get extra param.
     * Note: this method was redefined in v2.4, which may affect older client code.
     * @return the extra params.
     */
    const std::unordered_map<std::string, std::string>&
    ExtraParams() const;

    /**
     * @brief Construct a new IndexDesc from Json object
     * @param json Json string for parse
     */
    Status
    ExtraParamsFromJson(std::string json);

    /**
     * @brief Set index state.
     * @param [in] code the code.
     */
    Status
    SetStateCode(const milvus::IndexStateCode& code);

    /**
     * @brief Get index state.
     * @return the state code.
     */
    milvus::IndexStateCode
    StateCode() const;

    /**
     * @brief Set index failed reason.
     * @param [in] reason the reason.
     */
    Status
    SetFailReason(const std::string& reason);

    /**
     * @brief Get index failed reason.
     * @return the fail reason.
     */
    std::string
    FailReason() const;

    /**
     * @brief Set number of indexed rows.
     * @param [in] rows the rows.
     */
    Status
    SetIndexedRows(int64_t rows);

    /**
     * @brief Get number of indexed rows.
     * Note that indexed rows could be larger than total rows, because some segments will be reindexed
     * after compaction.
     * @return the indexed rows.
     */
    int64_t
    IndexedRows() const;

    /**
     * @brief Set number of total rows.
     * @param [in] rows the rows.
     */
    Status
    SetTotalRows(int64_t rows);

    /**
     * @brief Get number of total rows.
     * @return the total rows.
     */
    int64_t
    TotalRows() const;

    /**
     * @brief Set number of pending unindexed rows.
     * @param [in] rows the rows.
     */
    Status
    SetPendingRows(int64_t rows);

    /**
     * @brief Get number of pending unindexed rows.
     * @return the pending rows.
     */
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
