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

namespace milvus {

/**
 * @brief Arguments for MilvusClient::Search().
 */
class SearchArguments {
 public:
    /**
     * @brief Get the target db name
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set target db name, default is empty, means use the db name of MilvusClient
     */
    Status
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Get name of the target collection
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of this collection, cannot be empty
     */
    Status
    SetCollectionName(std::string collection_name);

    /**
     * @brief Get partition names
     */
    const std::set<std::string>&
    PartitionNames() const;

    /**
     * @brief Specify partition name to control search scope, the name cannot be empty
     */
    Status
    AddPartitionName(std::string partition_name);

    /**
     * @brief Get output field names
     */
    const std::set<std::string>&
    OutputFields() const;

    /**
     * @brief Specify output field names to return field data, the name cannot be empty
     */
    Status
    AddOutputField(std::string field_name);

    /**
     * @brief Get filter expression
     */
    const std::string&
    Filter() const;

    /**
     * @brief Set filter expression
     */
    Status
    SetFilter(std::string filter);

    /**
     * @brief Get target vectors
     */
    FieldDataPtr
    TargetVectors() const;

    /**
     * @brief Add a binary vector to search
     */
    Status
    AddBinaryVector(std::string field_name, const std::vector<uint8_t>& vector);

    /**
     * @brief Add a binary vector to search
     */
    Status
    AddBinaryVector(std::string field_name, const BinaryVecFieldData::ElementT& vector);

    /**
     * @brief Add a float vector to search
     */
    Status
    AddFloatVector(std::string field_name, const FloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a sparse vector to search
     */
    Status
    AddSparseVector(std::string field_name, const SparseFloatVecFieldData::ElementT& vector);

    /**
     * @brief Get anns field name
     */
    std::string
    AnnsField() const;

    /**
     * @brief Get search limit(topk)
     */
    int64_t
    Limit() const;

    /**
     * @brief Set search limit(topk)
     * Note: this value is stored in the ExtraParams
     */
    Status
    SetLimit(int64_t limit);

    /**
     * @brief Get offset value.
     */
    int64_t
    Offset() const;

    /**
     * @brief Set offset value
     * Note: this value is stored in the ExtraParams
     */
    Status
    SetOffset(int64_t offset);

    /**
     * @brief Specifies the decimal place of the returned results.
     * Note: this value is stored in the ExtraParams
     */
    Status
    SetRoundDecimal(int round_decimal);

    /**
     * @brief Get the decimal place of the returned results
     */
    int
    RoundDecimal() const;

    /**
     * @brief Specifies the metric type
     */
    Status
    SetMetricType(::milvus::MetricType metric_type);

    /**
     * @brief Get the metric type
     */
    ::milvus::MetricType
    MetricType() const;

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
     * @brief Validate for search arguments and get name of the target anns field
     * Note: in v2.4+, a collection can have one or more vector fields. If a collection has
     * only one vector field, users can set an empty name in the AddTargetVector(),
     * the server can know the vector field name.
     * But if the collection has multiple vector fields, users need to provide a non-empty name
     * in the AddTargetVector() method, and if users call AddTargetVector() mutiple times, he must
     * ensure that the name is the same, otherwise the Validate() method will return an error.
     * The Validate() method is called before Search().
     */
    Status
    Validate() const;

    /**
     * @brief Get range radius
     * @return
     */
    float
    Radius() const;

    /**
     * @brief Set range radius
     * Note: this value is stored in the ExtraParams
     * @return
     */
    Status
    SetRadius(float value);

    /**
     * @brief Get range filter
     * @return
     */
    float
    RangeFilter() const;

    /**
     * @brief Set range filter
     * Note: this value is stored in the ExtraParams
     * @return
     */
    Status
    SetRangeFilter(float value);

    /**
     * @brief Set range radius
     * @param range_filter while radius sets the outer limit of the search, range_filter can be optionally used to
     * define an inner boundary, creating a distance range within which vectors must fall to be considered matches.
     * @param radius defines the outer boundary of your search space. Only vectors that are within this distance from
     * the query vector are considered potential matches.
     */
    Status
    SetRange(float range_filter, float radius);

    /**
     * @brief Get consistency level
     */
    ConsistencyLevel
    GetConsistencyLevel() const;

    /**
     * @brief Set consistency level
     */
    Status
    SetConsistencyLevel(const ConsistencyLevel& level);

    /**
     * @brief Get ignore growing segments
     */
    bool
    IgnoreGrowing() const;

    /**
     * @brief Set ignore growing segments
     */
    Status
    SetIgnoreGrowing(bool ignore_growing);

    ///////////////////////////////////////////////////////////////////////////////////////
    // deprecated methods
    /**
     * @brief Get filter expression
     * @deprecated replaced by Filter()
     */
    const std::string&
    Expression() const;

    /**
     * @brief Set filter expression
     * @deprecated replaced by SetFilter()
     */
    Status
    SetExpression(std::string expression);

    /**
     * @brief Specify search limit, AKA topk
     * @deprecated replaced by SetLimit()
     */
    Status
    SetTopK(int64_t topk);

    /**
     * @brief Get Top K
     * @deprecated replaced by Limit()
     */
    int64_t
    TopK() const;

    /**
     * @brief Get nprobe
     * @deprecated replaced by ExtraParams()
     */
    int64_t
    Nprobe() const;

    /**
     * @brief Set nprobe
     * @deprecated replaced by SetExtraParams()
     */
    Status
    SetNprobe(int64_t nlist);

    /**
     * @brief Add a binary vector to search
     * @deprecated replaced by AddBinaryVector
     */
    Status
    AddTargetVector(std::string field_name, const std::string& vector);

    /**
     * @brief Add a binary vector to search with uint8_t vectors
     * @deprecated replaced by AddBinaryVector
     */
    Status
    AddTargetVector(std::string field_name, const std::vector<uint8_t>& vector);

    /**
     * @brief Add a binary vector to search
     * @deprecated replaced by AddBinaryVector
     */
    Status
    AddTargetVector(std::string field_name, std::string&& vector);

    /**
     * @brief Add a float vector to search
     * @deprecated replaced by AddFloatVector
     */
    Status
    AddTargetVector(std::string field_name, const FloatVecFieldData::ElementT& vector);

    /**
     * @brief Add a float vector to search
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
     * time. \n
     *
     * Default value is 0, server executes search on a full data view.
     * @deprecated Deprecated in 2.4, replaced by ConsistencyLevel
     */
    Status
    SetTravelTimestamp(uint64_t timestamp);

    /**
     * @brief Get guarantee timestamp.
     * @deprecated Deprecated in 2.4, replaced by ConsistencyLevel, this value is not used anymore
     */
    uint64_t
    GuaranteeTimestamp() const;

    /**
     * @brief Instructs server to see insert/delete operations performed before a provided timestamp. \n
     * If no such timestamp is specified, the server will wait for the latest operation to finish and search. \n
     *
     * Note: The timestamp is not an absolute timestamp, it is a hybrid value combined by UTC time and internal flags.
     * \n We call it TSO, for more information please refer to: \n
     * https://github.com/milvus-io/milvus/blob/master/docs/design_docs/milvus_hybrid_ts_en.md.
     * You can get a TSO from insert/delete results. Use an operation's TSO to set this parameter,
     * the server will execute search after this operation is finished. \n
     *
     * Default value is 1, server executes search immediately.
     * @deprecated Deprecated in 2.4, replaced by ConsistencyLevel
     */
    Status
    SetGuaranteeTimestamp(uint64_t timestamp);
    ///////////////////////////////////////////////////////////////////////////////////////
 private:
    Status verifyVectorType(DataType) const;

 private:
    std::string db_name_;
    std::string collection_name_;
    std::set<std::string> partition_names_;
    std::set<std::string> output_field_names_;
    std::string filter_expression_;

    FieldDataPtr target_vectors_;

    std::set<std::string> output_fields_;
    std::unordered_map<std::string, std::string> extra_params_;

    uint64_t travel_timestamp_{0};

    int64_t limit_{10};
    int round_decimal_{-1};

    ::milvus::MetricType metric_type_{::milvus::MetricType::DEFAULT};

    // ConsistencyLevel::NONE means using collection's default level
    ConsistencyLevel consistency_level_{ConsistencyLevel::NONE};

    bool ignore_growing_{false};
};

}  // namespace milvus
