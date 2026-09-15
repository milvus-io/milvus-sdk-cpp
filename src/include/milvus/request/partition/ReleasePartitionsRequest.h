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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::ReleasePartitions()
 */
class MILVUS_SDK_API ReleasePartitionsRequest {
 public:
    /**
     * @brief Constructor
     */
    ReleasePartitionsRequest() = default;

    /**
     * @brief Database name in which the collection is created.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name in which the collection is created.
     * @param [in] db_name the DB name.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name in which the collection is created.
     * @param [in] db_name the DB name.
     */
    ReleasePartitionsRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Name of the collection.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of the collection.
     * @param [in] collection_name the collection name.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set name of the collection.
     * @param [in] collection_name the collection name.
     */
    ReleasePartitionsRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Name of the partitions.
     * @return the partition names.
     */
    const std::set<std::string>&
    PartitionNames() const;

    /**
     * @brief Set name of the partitions.
     * @param [in] partition_names the partition names.
     */
    void
    SetPartitionNames(const std::set<std::string>& partition_names);

    /**
     * @brief Set new name of the partitions.
     * @param [in] partition_names the partition names.
     */
    ReleasePartitionsRequest&
    WithPartitionNames(const std::set<std::string>& partition_names);

    /**
     * @brief Add a partition to be released.
     * @param [in] partition_name the partition name.
     */
    ReleasePartitionsRequest&
    AddPartitionName(const std::string& partition_name);

 private:
    std::string db_name_;
    std::string collection_name_;
    std::set<std::string> partition_names_;
};

}  // namespace milvus
