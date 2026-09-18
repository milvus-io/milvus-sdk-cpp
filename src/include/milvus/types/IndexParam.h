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

#include <string>
#include <unordered_map>

#include "../Status.h"
#include "IndexType.h"
#include "MetricType.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Parameters used to create an index on a collection field.
 * Used by MilvusClientV2::CreateIndex().
 */
class MILVUS_SDK_API IndexParam {
 public:
    /**
     * @brief Construct a new IndexParam object.
     */
    IndexParam();

    /**
     * @brief Construct a new IndexParam object.
     *
     * @param field_name field name which the index belong to
     * @param index_name index name
     * @param index_type index type see IndexType
     * @param metric_type metric type see MetricType, no need to set this for scalar field index
     */
    IndexParam(std::string field_name, std::string index_name, milvus::IndexType index_type,
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
     * @brief Index name.
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
     * @param [in] key the key.
     * @param [in] value the value.
     */
    Status
    AddExtraParam(const std::string& key, const std::string& value);

    /**
     * @brief Get extra params.
     * @return the extra params.
     */
    const std::unordered_map<std::string, std::string>&
    ExtraParams() const;

    /**
     * @brief Construct extra params from a Json object.
     * @param json Json string for parse
     */
    Status
    ExtraParamsFromJson(std::string json);

 private:
    std::string field_name_;
    std::string index_name_;
    milvus::MetricType metric_type_{milvus::MetricType::INVALID};
    milvus::IndexType index_type_{milvus::IndexType::INVALID};
    std::unordered_map<std::string, std::string> extra_params_;
};

}  // namespace milvus
