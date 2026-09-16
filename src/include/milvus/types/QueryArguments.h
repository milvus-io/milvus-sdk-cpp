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
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <set>
#include <string>
#include <unordered_map>

#include "../Status.h"
#include "ConsistencyLevel.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Arguments for MilvusClient::Query().
 */
class MILVUS_SDK_API QueryArguments {
 public:
    virtual ~QueryArguments() = default;

    /**
     * @brief Get the target db name.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set target db name, default is empty, means use the db name of MilvusClient.
     * @param [in] db_name the DB name.
     */
    Status
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Get name of the target collection.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of this collection, cannot be empty.
     * @param [in] collection_name the collection name.
     */
    Status
    SetCollectionName(std::string collection_name);

    /**
     * @brief Get partition names.
     * @return the partition names.
     */
    const std::set<std::string>&
    PartitionNames() const;

    /**
     * @brief Specify partition name to control query scope, the name cannot be empty.
     * @param [in] partition_name the partition name.
     */
    Status
    AddPartitionName(std::string partition_name);

    /**
     * @brief Get output field names.
     * @return the output fields.
     */
    const std::set<std::string>&
    OutputFields() const;
    /**
     * @brief Specify output field names to return field data, the name cannot be empty.
     * @param [in] field_name the field name.
     */
    Status
    AddOutputField(std::string field_name);

    /**
     * @brief Get filter expression.
     * @return the filter.
     */
    const std::string&
    Filter() const;

    /**
     * @brief Set filter expression.
     * @param [in] filter the filter.
     */
    Status
    SetFilter(std::string filter);

    /**
     * @brief Add a filter template.
     * Expression template, to improve expression parsing performance in complicated list.
     * Assume user has a filter = "pk > 3 and city in ["beijing", "shanghai", ......]
     * The long list of city will increase the time cost to parse this expression.
     * So, we provide filterTemplate for this purpose, user can set filter like this:
     *     filter = "pk > {age} and city in {city}"
     *     filterTemplate = {"age": 3, "city": ["beijing", "shanghai", ......]}
     * Valid value of a template can be:
     *     boolean, numeric, string, array.
     * @param [in] key the key.
     * @param [in] filter_template the filter template.
     */
    Status
    AddFilterTemplate(std::string key, const nlohmann::json& filter_template);

    /**
     * @brief Get filter templates.
     * @return the filter templates.
     */
    const std::unordered_map<std::string, nlohmann::json>&
    FilterTemplates() const;

    /**
     * @brief Get limit value.
     * @return the limit.
     */
    int64_t
    Limit() const;

    /**
     * @brief Set limit value, only avaiable when expression is empty.
     * Note: this value is stored in the ExtraParams.
     * @param [in] limit the limit.
     */
    Status
    SetLimit(int64_t limit);

    /**
     * @brief Get offset value.
     * @return the offset.
     */
    int64_t
    Offset() const;

    /**
     * @brief Set offset value, only avaiable when expression is empty.
     * Note: this value is stored in the ExtraParams.
     * @param [in] offset the offset.
     */
    Status
    SetOffset(int64_t offset);

    /**
     * @brief Get ignore growing segments.
     * @return the ignore growing.
     */
    bool
    IgnoreGrowing() const;

    /**
     * @brief Set ignore growing segments.
     * @param [in] ignore_growing the ignore growing.
     */
    Status
    SetIgnoreGrowing(bool ignore_growing);

    /**
     * @brief Add extra param.
     * @param [in] key the key.
     * @param [in] value the value.
     */
    Status
    AddExtraParam(const std::string& key, const std::string& value);

    /**
     * @brief Get extra param.
     * @return the extra params.
     */
    const std::unordered_map<std::string, std::string>&
    ExtraParams() const;

    /**
     * @brief Get consistency level.
     * @return the consistency level.
     */
    ConsistencyLevel
    GetConsistencyLevel() const;

    /**
     * @brief Set consistency level.
     * @param [in] level the level.
     */
    Status
    SetConsistencyLevel(const ConsistencyLevel& level);

    ///////////////////////////////////////////////////////////////////////////////////////
    // deprecated methods
    /**
     * @brief Get filter expression.
     * Can be empty if Limit() is zero, else must be non-empty.
     * @deprecated replaced by Filter()
     * @return the expression.
     */
    const std::string&
    Expression() const;

    /**
     * @brief Set filter expression.
     * Can be empty if Limit() is zero, else must be non-empty.
     * @deprecated replaced by SetFilter()
     * @param [in] expression the expression.
     */
    Status
    SetExpression(std::string expression);

    /**
     * @brief Get travel timestamp.
     * @deprecated Deprecated in 2.4, replaced by ConsistencyLevel.
     * @return the travel timestamp.
     */
    uint64_t
    TravelTimestamp() const;
    /**
     * @brief Specify an absolute timestamp in a query to get results based on a data view at a specified point
     * in time.
     * Default value is 0, server executes query on a full data view.
     * @deprecated Deprecated in 2.4, replaced by ConsistencyLevel.
     * @param [in] timestamp the timestamp.
     */
    Status
    SetTravelTimestamp(uint64_t timestamp);

    /**
     * @brief Get guarantee timestamp.
     * @deprecated Deprecated in 2.4, replaced by ConsistencyLevel, this value is not used anymore.
     * @return the guarantee timestamp.
     */
    uint64_t
    GuaranteeTimestamp() const;

    /**
     * @brief Instructs server to see insert/delete operations performed before a provided timestamp.
     * If no such timestamp is specified, the server will wait for the latest operation to finish and query.
     *
     * Note: The timestamp is not an absolute timestamp, it is a hybrid value combined by UTC time and internal flags.
     * We call it TSO, for more information please refer to:
     * https://github.com/milvus-io/milvus/blob/master/docs/design_docs/milvus_hybrid_ts_en.md.
     * You can get a TSO from insert/delete results. Use an operation's TSO to set this parameter, the server will
     * execute query after this operation is finished.
     *
     * Default value is 1, server executes search immediately.
     * @deprecated Deprecated in 2.4, replaced by ConsistencyLevel
     * @param [in] timestamp the timestamp.
     */
    Status
    SetGuaranteeTimestamp(uint64_t timestamp);
    ///////////////////////////////////////////////////////////////////////////////////////
 private:
    std::string db_name_;
    std::string collection_name_;
    std::set<std::string> partition_names_;
    std::set<std::string> output_field_names_;
    std::string filter_expression_;
    std::unordered_map<std::string, nlohmann::json> filter_templates_;

    std::unordered_map<std::string, std::string> extra_params_;

    uint64_t travel_timestamp_{0};

    // ConsistencyLevel::NONE means using collection's default level
    ConsistencyLevel consistency_level_{ConsistencyLevel::NONE};
};

}  // namespace milvus
