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

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "../Status.h"
#include "ConsistencyLevel.h"
#include "FieldData.h"
#include "Function.h"
#include "SubSearchRequest.h"

namespace milvus {

/**
 * @brief Arguments for MilvusClient::HybridSearch().
 */
class HybridSearchArguments {
 public:
    /**
     * @brief Get the target db name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set target db name, default is empty, means use the db name of MilvusClient.
     */
    Status
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Get name of the target collection.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of this collection, cannot be empty.
     */
    Status
    SetCollectionName(std::string collection_name);

    /**
     * @brief Get partition names.
     */
    const std::set<std::string>&
    PartitionNames() const;

    /**
     * @brief Specify partition name to control search scope, the name cannot be empty.
     */
    Status
    AddPartitionName(std::string partition_name);

    /**
     * @brief Get output field names.
     */
    const std::set<std::string>&
    OutputFields() const;

    /**
     * @brief Specify output field names to return field data, the name cannot be empty.
     */
    Status
    AddOutputField(std::string field_name);

    /**
     * @brief Get search limit(topk).
     */
    int64_t
    Limit() const;

    /**
     * @brief Set search limit(topk).
     * Note: this value is stored in the ExtraParams.
     */
    Status
    SetLimit(int64_t limit);

    /**
     * @brief Get offset value.
     */
    int64_t
    Offset() const;

    /**
     * @brief Set offset value.
     * Note: this value is stored in the ExtraParams.
     */
    Status
    SetOffset(int64_t offset);

    /**
     * @brief Get the decimal place of the returned results.
     */
    int
    RoundDecimal() const;

    /**
     * @brief Specifies the decimal place of the returned results.
     * Note: this value is stored in the ExtraParams.
     */
    Status
    SetRoundDecimal(int round_decimal);

    /**
     * @brief Get consistency level.
     */
    ConsistencyLevel
    GetConsistencyLevel() const;

    /**
     * @brief Set consistency level.
     */
    Status
    SetConsistencyLevel(const ConsistencyLevel& level);

    /**
     * @brief Get ignore growing segments.
     */
    bool
    IgnoreGrowing() const;

    /**
     * @brief Set ignore growing segments.
     */
    Status
    SetIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Get sub search requests.
     */
    const std::vector<SubSearchRequestPtr>&
    SubRequests() const;

    /**
     * @brief Add sub search request.
     */
    Status
    AddSubRequest(const SubSearchRequestPtr& request);

    /**
     * @brief Get rerank.
     */
    FunctionPtr
    Rerank() const;

    /**
     * @brief Set rerank, only accept RERANK function type.
     */
    Status
    SetRerank(const FunctionPtr& rerank);

    /**
     * @brief Get group by field name.
     */
    std::string
    GroupByField() const;

    /**
     * @brief Set group by field name.
     */
    Status
    SetGroupByField(const std::string& field_name);

    /**
     * @brief Get size of group by.
     */
    uint64_t
    GroupSize() const;

    /**
     * @brief Set size of group by.
     */
    Status
    SetGroupSize(uint64_t group_size);

    /**
     * @brief Get the flag whether to strict group size.
     */
    bool
    StrictGroupSize() const;

    /**
     * @brief Set the flag whether to strict group size.
     */
    Status
    SetStrictGroupSize(bool strict_group_size);

    /**
     * @brief Add extra param.
     */
    Status
    AddExtraParam(const std::string& key, const std::string& value);

    /**
     * @brief Get extra param.
     */
    const std::unordered_map<std::string, std::string>&
    ExtraParams() const;

    /**
     * @brief Validate for search arguments.
     * MilvusClient calls this method internally, users no need to mamually call it.
     */
    Status
    Validate() const;

 private:
    std::string db_name_;
    std::string collection_name_;
    std::set<std::string> partition_names_;
    std::set<std::string> output_field_names_;
    std::vector<SubSearchRequestPtr> sub_requests_;
    FunctionPtr function_;

    int64_t limit_{10};
    std::unordered_map<std::string, std::string> extra_params_;

    // ConsistencyLevel::NONE means using collection's default level
    ConsistencyLevel consistency_level_{ConsistencyLevel::NONE};
};

}  // namespace milvus
