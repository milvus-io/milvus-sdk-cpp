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

#include "./IndexRequestBase.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::DescribeIndex()
 */
class MILVUS_SDK_API DescribeIndexRequest : public IndexRequestBase<DescribeIndexRequest> {
 public:
    /**
     * @brief Constructor
     */
    DescribeIndexRequest() = default;

    /**
     * @brief Name of the field.
     * @return the field name.
     */
    const std::string&
    FieldName() const;

    /**
     * @brief Set name of the field.
     * @param [in] field_name the field name.
     */
    void
    SetFieldName(const std::string& field_name);

    /**
     * @brief Set name of the field.
     * @param [in] field_name the field name.
     */
    DescribeIndexRequest&
    WithFieldName(const std::string& field_name);

    /**
     * @brief Name of the index.
     * @return the index name.
     */
    const std::string&
    IndexName() const;

    /**
     * @brief Set name of the index.
     * Note: if both field_name and index_name are specified, it will use index name firstly.
     * @param [in] index_name the index name.
     */
    void
    SetIndexName(const std::string& index_name);

    /**
     * @brief Set name of the index.
     * Note: if both field_name and index_name are specified, it will use index name firstly.
     * @param [in] index_name the index name.
     */
    DescribeIndexRequest&
    WithIndexName(const std::string& index_name);

    /**
     * @brief Timestamp to skip segments.
     * @return the timestamp.
     */
    int64_t
    Timestamp() const;

    /**
     * @brief Only check segments generated before this timestamp. all the segments will be checked if this value is
     * zero.
     * @param [in] ts the ts.
     */
    void
    SetTimestamp(int64_t ts);

    /**
     * @brief Only check segments generated before this timestamp. all the segments will be checked if this value is
     * zero.
     * @param [in] ts the ts.
     */
    DescribeIndexRequest&
    WithTimestamp(int64_t ts);

 private:
    std::string field_name_;
    std::string index_name_;
    int64_t timestamp_{0};
};

}  // namespace milvus
