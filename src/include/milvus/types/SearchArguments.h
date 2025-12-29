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

#include "../Status.h"
#include "ConsistencyLevel.h"
#include "Constants.h"
#include "FieldData.h"
#include "MetricType.h"
#include "SearchRequestBase.h"

namespace milvus {

/**
 * @brief Arguments for MilvusClient::Search().
 */
class SearchArguments : public SearchRequestBase {
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
     * Note: this value is stored in the ExtraParams.
     */
    Status
    SetIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Get group by field name.
     */
    std::string
    GroupByField() const;

    /**
     * @brief Set group by field name.
     * Note: this value is stored in the ExtraParams.
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
     * Note: this value is stored in the ExtraParams.
     */
    Status
    SetGroupSize(uint64_t group_size);

    /**
     * @brief Get the flag whether to strict group size.
     */
    uint64_t
    StrictGroupSize() const;

    /**
     * @brief Set the flag whether to strict group size.
     * Note: this value is stored in the ExtraParams.
     */
    Status
    SetStrictGroupSize(bool strict_group_size);

    ///////////////////////////////////////////////////////////////////////////////////////
    // deprecated methods
    /**
     * @brief Get filter expression.
     * @deprecated replaced by Filter()
     */
    const std::string&
    Expression() const;

    /**
     * @brief Set filter expression.
     * @deprecated replaced by SetFilter()
     */
    Status
    SetExpression(std::string expression);

    /**
     * @brief Specify search limit, AKA topk.
     * @deprecated replaced by SetLimit()
     */
    Status
    SetTopK(int64_t topk);

    /**
     * @brief Get Top K.
     * @deprecated replaced by Limit()
     */
    int64_t
    TopK() const;

    /**
     * @brief Get nprobe.
     * @deprecated replaced by ExtraParams()
     */
    int64_t
    Nprobe() const;

    /**
     * @brief Set nprobe.
     * @deprecated replaced by AddExtraParam()
     */
    Status
    SetNprobe(int64_t nlist);

    /**
     * @brief Add a binary vector to search.
     * @deprecated replaced by AddBinaryVector
     */
    Status
    AddTargetVector(std::string field_name, const std::string& vector);

    /**
     * @brief Add a binary vector to search with uint8_t vectors.
     * @deprecated replaced by AddBinaryVector
     */
    Status
    AddTargetVector(std::string field_name, const std::vector<uint8_t>& vector);

    /**
     * @brief Add a binary vector to search.
     * @deprecated replaced by AddBinaryVector
     */
    Status
    AddTargetVector(std::string field_name, std::string&& vector);

    /**
     * @brief Add a float vector to search.
     * @deprecated replaced by AddFloatVector
     */
    Status
    AddTargetVector(std::string field_name, const FloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a float vector to search.
     * @deprecated replaced by AddFloatVector
     */
    Status
    AddTargetVector(std::string field_name, FloatVecFieldData::ElementT&& vector);

    /**
     * @brief Get travel timestamp.
     * @deprecated Deprecated in 2.4, replaced by ConsistencyLevel
     */
    uint64_t
    TravelTimestamp() const;

    /**
     * @brief Specify an absolute timestamp in a search to get results based on a data view at a specified point in
     * time.
     * Default value is 0, server executes search on a full data view.
     * @deprecated Deprecated in 2.4, replaced by ConsistencyLevel
     */
    Status
    SetTravelTimestamp(uint64_t timestamp);

    /**
     * @brief Get guarantee timestamp.
     * @deprecated Deprecated in 2.4, replaced by ConsistencyLevel, this value is not used anymore.
     */
    uint64_t
    GuaranteeTimestamp() const;

    /**
     * @brief Instructs server to see insert/delete operations performed before a provided timestamp.
     * If no such timestamp is specified, the server will wait for the latest operation to finish and search.
     *
     * Note: The timestamp is not an absolute timestamp, it is a hybrid value combined by UTC time and internal flags.
     * We call it TSO, for more information please refer to:
     * https://github.com/milvus-io/milvus/blob/master/docs/design_docs/milvus_hybrid_ts_en.md.
     * You can get a TSO from insert/delete results. Use an operation's TSO to set this parameter,
     * the server will execute search after this operation is finished.
     *
     * Default value is 1, server executes search immediately.
     * @deprecated Deprecated in 2.4, replaced by ConsistencyLevel.
     */
    Status
    SetGuaranteeTimestamp(uint64_t timestamp);

 private:
    std::string db_name_;
    std::string collection_name_;
    std::set<std::string> partition_names_;
    std::set<std::string> output_field_names_;

    // ConsistencyLevel::NONE means using collection's default level
    ConsistencyLevel consistency_level_{ConsistencyLevel::NONE};
};

}  // namespace milvus
