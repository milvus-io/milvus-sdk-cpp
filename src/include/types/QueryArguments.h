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

#include "Status.h"

namespace milvus {

/**
 * @brief Arguments for Query().
 */
class QueryArguments {
 public:
    QueryArguments() = default;

    /**
     * @brief Get name of the target collection
     */
    const std::string&
    CollectionName() const {
        return collection_name_;
    }

    /**
     * @brief Set name of this collection, cannot be empty
     */
    Status
    SetCollectionName(const std::string& collection_name) {
        if (collection_name.empty()) {
            return {StatusCode::INVALID_AGUMENT, "Collection name cannot be empty!"};
        }
        collection_name_ = collection_name;
        return Status::OK();
    }

    /**
     * @brief Get partition names
     */
    const std::set<std::string>&
    PartitionNames() const {
        return partition_names_;
    }

    /**
     * @brief Specify partition name to control query scope, the name cannot be empty
     */
    Status
    AddPartitionName(const std::string& partition_name) {
        if (partition_name.empty()) {
            return {StatusCode::INVALID_AGUMENT, "Partition name cannot be empty!"};
        }

        partition_names_.insert(partition_name);
        return Status::OK();
    }

    /**
     * @brief Get output field names
     */
    const std::set<std::string>&
    OutputFields() const {
        return output_field_names_;
    }

    /**
     * @brief Specify output field names to return field data, the name cannot be empty
     */
    Status
    AddOutputField(const std::string& field_name) {
        if (field_name.empty()) {
            return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
        }

        output_field_names_.insert(field_name);
        return Status::OK();
    }

    /**
     * @brief Get filter expression
     */
    const std::string&
    Expression() const {
        return filter_expression_;
    }

    /**
     * @brief Set filter expression, the expression cannot be empty
     */
    Status
    SetExpression(const std::string& expression) {
        if (expression.empty()) {
            return {StatusCode::INVALID_AGUMENT, "Filter expression cannot be empty!"};
        }

        filter_expression_ = expression;
        return Status::OK();
    }

    /**
     * @brief Get travel timestamp.
     */
    uint64_t
    TravelTimestamp() const {
        return travel_timestamp_;
    }

    /**
     * @brief Specify a timestamp in a search to get results based on a data view at a specified point in time.
     */
    Status
    SetTravelTimestamp(uint64_t timestamp) {
        travel_timestamp_ = timestamp;
        return Status::OK();
    }

    /**
     * @brief Get guarantee timestamp.
     */
    uint64_t
    GuaranteeTimestamp() const {
        return guarantee_timestamp_;
    }

    /**
     * @brief Specify a timestamp so that the server can see all operations performed before the provided timestamp.
     * If no such timestamp is provided, the server will search all operations performed to date.
     */
    Status
    SetGuaranteeTimestamp(uint64_t timestamp) {
        guarantee_timestamp_ = timestamp;
        return Status::OK();
    }

 private:
    std::string collection_name_;
    std::set<std::string> partition_names_;
    std::set<std::string> output_field_names_;
    std::string filter_expression_;

    std::set<std::string> output_fields_;

    uint64_t travel_timestamp_ = 0;
    uint64_t guarantee_timestamp_ = 0;
};

}  // namespace milvus
