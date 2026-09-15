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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::RestoreSnapshot()
 */
class MILVUS_SDK_API RestoreSnapshotRequest {
 public:
    /**
     * @brief Constructor
     */
    RestoreSnapshotRequest() = default;

    /**
     * @brief Get the name of the snapshot to restore.
     * @return the snapshot name.
     */
    const std::string&
    SnapshotName() const;

    /**
     * @brief Set the name of the snapshot to restore.
     * @param [in] snapshot_name the snapshot name.
     */
    void
    SetSnapshotName(const std::string& snapshot_name);

    /**
     * @brief Set the name of the snapshot to restore.
     * @param [in] snapshot_name the snapshot name.
     */
    RestoreSnapshotRequest&
    WithSnapshotName(const std::string& snapshot_name);

    /**
     * @brief Get the source database name.
     * @return the source database name.
     */
    const std::string&
    SourceDatabaseName() const;

    /**
     * @brief Set the source database name.
     * @param [in] db_name the DB name.
     */
    void
    SetSourceDatabaseName(const std::string& db_name);

    /**
     * @brief Set the source database name.
     * @param [in] db_name the DB name.
     */
    RestoreSnapshotRequest&
    WithSourceDatabaseName(const std::string& db_name);

    /**
     * @brief Get the source collection name.
     * @return the source collection name.
     */
    const std::string&
    SourceCollectionName() const;

    /**
     * @brief Set the source collection name.
     * @param [in] collection_name the collection name.
     */
    void
    SetSourceCollectionName(const std::string& collection_name);

    /**
     * @brief Set the source collection name.
     * @param [in] collection_name the collection name.
     */
    RestoreSnapshotRequest&
    WithSourceCollectionName(const std::string& collection_name);

    /**
     * @brief Get the target database name.
     * @return the target database name.
     */
    const std::string&
    TargetDatabaseName() const;

    /**
     * @brief Set the target database name.
     * @param [in] db_name the DB name.
     */
    void
    SetTargetDatabaseName(const std::string& db_name);

    /**
     * @brief Set the target database name.
     * @param [in] db_name the DB name.
     */
    RestoreSnapshotRequest&
    WithTargetDatabaseName(const std::string& db_name);

    /**
     * @brief Get the target collection name.
     * @return the target collection name.
     */
    const std::string&
    TargetCollectionName() const;

    /**
     * @brief Set the target collection name.
     * @param [in] collection_name the collection name.
     */
    void
    SetTargetCollectionName(const std::string& collection_name);

    /**
     * @brief Set the target collection name.
     * @param [in] collection_name the collection name.
     */
    RestoreSnapshotRequest&
    WithTargetCollectionName(const std::string& collection_name);

 private:
    std::string snapshot_name_;
    std::string source_db_name_;
    std::string source_collection_name_;
    std::string target_db_name_;
    std::string target_collection_name_;
};

}  // namespace milvus
