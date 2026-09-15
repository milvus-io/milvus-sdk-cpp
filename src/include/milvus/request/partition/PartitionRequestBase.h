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

namespace milvus {

/**
 * @brief Base class of partition requests except LoadPartitionsRequest and ListPartitionsRequest.
 */
template <typename T>
class PartitionRequestBase {
 protected:
    /**
     * @brief Constructor
     */
    PartitionRequestBase() = default;

 public:
    /**
     * @brief Get the target db name
     * @return the database name.
     */
    const std::string&
    DatabaseName() const {
        return db_name_;
    }

    /**
     * @brief Set target db name, use default database if it is empty.
     * @param [in] db_name the DB name.
     */
    void
    SetDatabaseName(const std::string& db_name) {
        db_name_ = db_name;
    }

    /**
     * @brief Set target db name, use default database if it is empty.
     * @param [in] db_name the DB name.
     */
    T&
    WithDatabaseName(const std::string& db_name) {
        SetDatabaseName(db_name);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Get the collection name.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const {
        return collection_name_;
    }

    /**
     * @brief Set the collection name.
     * @param [in] collection_name the collection name.
     */
    void
    SetCollectionName(const std::string& collection_name) {
        collection_name_ = collection_name;
    }

    /**
     * @brief Set name of the collection.
     * @param [in] collection_name the collection name.
     */
    T&
    WithCollectionName(const std::string& collection_name) {
        SetCollectionName(collection_name);
        return static_cast<T&>(*this);
    }

    /**
     * @brief Set name of the partition.
     * @return the partition name.
     */
    const std::string&
    PartitionName() const {
        return partition_name_;
    }

    /**
     * @brief Set name of the partition.
     * @param [in] partition_name the partition name.
     */
    void
    SetPartitionName(const std::string& partition_name) {
        partition_name_ = partition_name;
    }

    /**
     * @brief Set name of the partition.
     * @param [in] partition_name the partition name.
     */
    T&
    WithPartitionName(const std::string& partition_name) {
        SetPartitionName(partition_name);
        return static_cast<T&>(*this);
    }

 protected:
    std::string db_name_;
    std::string collection_name_;
    std::string partition_name_;
};

}  // namespace milvus
