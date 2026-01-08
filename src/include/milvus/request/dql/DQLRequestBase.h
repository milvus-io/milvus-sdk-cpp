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
template <typename T>
class DQLRequestBase {
 protected:
    DQLRequestBase() = default;

 public:
    /**
     * @brief Get the target db name
     */
    const std::string&
    DatabaseName() const {
        return db_name_;
    }

    /**
     * @brief Set target db name, use default database if it is empty.
     */
    void
    SetDatabaseName(const std::string& db_name) {
        db_name_ = db_name;
    }

    /**
     * @brief Set target db name, use default database if it is empty.
     */
    T&
    WithDatabaseName(const std::string& db_name) {
        SetDatabaseName(db_name);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the collection name.
     */
    const std::string&
    CollectionName() const {
        return collection_name_;
    }

    /**
     * @brief Set the collection name.
     */
    void
    SetCollectionName(const std::string& collection_name) {
        collection_name_ = collection_name;
    }

    /**
     * @brief Set name of the collection.
     */
    T&
    WithCollectionName(const std::string& collection_name) {
        SetCollectionName(collection_name);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the partition names.
     */
    const std::set<std::string>&
    PartitionNames() const {
        return partition_names_;
    }

    /**
     * @brief Set the partition names.
     * If partition nemes are empty, will query in the entire collection.
     */
    void
    SetPartitionNames(std::set<std::string>&& partition_names) {
        partition_names_ = std::move(partition_names);
    }

    /**
     * @brief Set the partition names.
     * If partition nemes are empty, will query in the entire collection.
     */
    T&
    WithPartitionNames(std::set<std::string>&& partition_names) {
        SetPartitionNames(std::move(partition_names));
        return static_cast<T&>(*this);
    }

    /**
     * @brief Add a partition name.
     */
    T&
    AddPartitionName(const std::string& partition_name) {
        partition_names_.insert(partition_name);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the output field names.
     */
    const std::set<std::string>&
    OutputFields() const {
        return output_field_names_;
    }

    /**
     * @brief Set the output field names.
     */
    void
    SetOutputFields(std::set<std::string>&& output_field_names) {
        output_field_names_ = std::move(output_field_names);
    }

    /**
     * @brief Set the output field names.
     */
    T&
    WithOutputFields(std::set<std::string>&& output_field_names) {
        SetOutputFields(std::move(output_field_names));
        return static_cast<T&>(*this);
    }

    /**
     * @brief Add an output field.
     */
    T&
    AddOutputField(const std::string& output_field) {
        output_field_names_.insert(output_field);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the consistency level.
     */
    ::milvus::ConsistencyLevel
    GetConsistencyLevel() const {
        return consistency_level_;
    }

    /**
     * @brief Set the consistency level.
     */
    void
    SetConsistencyLevel(::milvus::ConsistencyLevel consistency_level) {
        consistency_level_ = consistency_level;
    }

    /**
     * @brief Set the consistency level.
     * Read the doc for more info: https://milvus.io/docs/consistency.md#Consistency-Level
     */
    T&
    WithConsistencyLevel(ConsistencyLevel consistency_level) {
        SetConsistencyLevel(consistency_level);
        return static_cast<T&>(*this);
    }

 private:
    std::string db_name_;
    std::string collection_name_;
    std::set<std::string> partition_names_;
    std::set<std::string> output_field_names_;
    ::milvus::ConsistencyLevel consistency_level_{ConsistencyLevel::NONE};
};

}  // namespace milvus
