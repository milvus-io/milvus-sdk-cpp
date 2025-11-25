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

namespace milvus {

/**
 * @brief Used by MilvusClientV2::DescribeIndex()
 */
class DescribeIndexRequest : public IndexRequestBase {
 public:
    /**
     * @brief Constructor
     */
    DescribeIndexRequest() = default;

    /**
     * @brief Set database name in which the collection is created.
     */
    DescribeIndexRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Set name of the collection.
     */
    DescribeIndexRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Name of the field.
     */
    const std::string&
    FieldName() const;

    /**
     * @brief Set name of the field.
     */
    void
    SetFieldName(const std::string& field_name);

    /**
     * @brief Set name of the field.
     */
    DescribeIndexRequest&
    WithFieldName(const std::string& field_name);

    /**
     * @brief Timestamp to skip segments.
     */
    int64_t
    Timestamp() const;

    /**
     * @brief Only check segments generated before this timestamp. all the segments will be checked if this value is
     * zero.
     */
    void
    SetTimestamp(int64_t ts);

    /**
     * @brief Only check segments generated before this timestamp. all the segments will be checked if this value is
     * zero.
     */
    DescribeIndexRequest&
    WithTimestamp(int64_t ts);

 private:
    std::string field_name_;
    int64_t timestamp_{0};
};

}  // namespace milvus
