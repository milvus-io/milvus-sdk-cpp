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
    RestoreSnapshotRequest() = default;

    const std::string&
    SnapshotName() const;

    void
    SetSnapshotName(const std::string& snapshot_name);

    RestoreSnapshotRequest&
    WithSnapshotName(const std::string& snapshot_name);

    const std::string&
    SourceDatabaseName() const;

    void
    SetSourceDatabaseName(const std::string& db_name);

    RestoreSnapshotRequest&
    WithSourceDatabaseName(const std::string& db_name);

    const std::string&
    SourceCollectionName() const;

    void
    SetSourceCollectionName(const std::string& collection_name);

    RestoreSnapshotRequest&
    WithSourceCollectionName(const std::string& collection_name);

    const std::string&
    TargetDatabaseName() const;

    void
    SetTargetDatabaseName(const std::string& db_name);

    RestoreSnapshotRequest&
    WithTargetDatabaseName(const std::string& db_name);

    const std::string&
    TargetCollectionName() const;

    void
    SetTargetCollectionName(const std::string& collection_name);

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
