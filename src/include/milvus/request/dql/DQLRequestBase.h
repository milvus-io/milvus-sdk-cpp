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

#include "../../types/ConsistencyLevel.h"

namespace milvus {

/**
 * @brief Base class for DQL requests.
 */
class DQLRequestBase {
 protected:
    DQLRequestBase() = default;

 public:
    /**
     * @brief Get the target db name
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set target db name, use default database if it is empty.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set target db name, use default database if it is empty.
     */
    DQLRequestBase&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Get the collection name
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set the collection name/
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Get the partition names/
     */
    const std::set<std::string>&
    PartitionNames() const;

    /**
     * @brief Set the partition names/
     * If partition nemes are empty, will query in the entire collection.
     */
    void
    SetPartitionNames(std::set<std::string>&& partition_names);

    /**
     * @brief Add a partition name.
     */
    void
    AddPartitionName(const std::string& partition_name);

    /**
     * @brief Get the output field names/
     */
    const std::set<std::string>&
    OutputFields() const;

    /**
     * @brief Set the output field names
     */
    void
    SetOutputFields(std::set<std::string>&& output_field_names);

    /**
     * @brief Add an output field.
     */
    void
    AddOutputField(const std::string& output_field);

    /**
     * @brief Get the consistency level
     */
    ::milvus::ConsistencyLevel
    GetConsistencyLevel() const;

    /**
     * @brief Set the consistency level
     */
    void
    SetConsistencyLevel(::milvus::ConsistencyLevel consistency_level);

 private:
    std::string db_name_;
    std::string collection_name_;
    std::set<std::string> partition_names_;
    std::set<std::string> output_field_names_;
    ::milvus::ConsistencyLevel consistency_level_{ConsistencyLevel::NONE};
};

}  // namespace milvus
