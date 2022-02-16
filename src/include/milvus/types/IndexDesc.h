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

namespace milvus {

/**
 * @brief Index description. Used by MilvusClient::CreateIndex() and MilvusClient::DescribeIndex().
 */
class IndexDesc {
 public:
    IndexDesc() = default;

    /**
     * @brief Constructor
     */
    IndexDesc(const std::string& field_name, const std::string& index_name, int64_t index_id,
              std::unordered_map<std::string, std::string> params)
        : field_name_{field_name}, index_name_{index_name}, index_id_{index_id}, params_{params} {
    }

    /**
     * @brief Filed name which the index belong to.
     */
    const std::string&
    FieldName() const {
        return field_name_;
    }

    /**
     * @brief Set Field name. Field name cannot be empty.
     */
    void
    SetFieldName(const std::string& field_name) {
        field_name_ = field_name;
    }

    /**
     * @brief Index name. Index name cannot be empty.
     */
    const std::string&
    IndexName() const {
        return index_name_;
    }

    /**
     * @brief Set name of the index. Reserved for funture feature: multiple indice in one field.  \n
     * Only avaiable for DescribeIndex(). No need to specify it for CreateIndex().
     */
    void
    SetIndexName(const std::string& index_name) {
        index_name_ = index_name;
    }

    /**
     * @brief Index ID.
     */
    int64_t
    IndexId() const {
        return index_id_;
    }

    /**
     * @brief Set internal id of the index. Reserved for funture feature: multiple indice in one field.
     * Only avaiable for DescribeIndex(). No need to specify it for CreateIndex().
     */
    void
    SetIndexId(int64_t index_id) {
        index_id_ = index_id;
    }

    /**
     * @brief Parameters of the index.
     */
    const std::unordered_map<std::string, std::string>&
    Params() const {
        return params_;
    }

    /**
     * @brief Set Parameters of the index.
     */
    void
    SetParams(const std::unordered_map<std::string, std::string>& params) {
        params_ = params;
    }

 private:
    std::string field_name_;
    std::string index_name_;
    int64_t index_id_ = 0;

    std::unordered_map<std::string, std::string> params_;
};

}  // namespace milvus
