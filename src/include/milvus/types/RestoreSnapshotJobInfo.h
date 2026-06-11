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
#include <string>

#include "milvus/Export.h"
#include "milvus/types/RestoreSnapshotState.h"

namespace milvus {

/**
 * @brief Restore snapshot job information.
 */
class MILVUS_SDK_API RestoreSnapshotJobInfo {
 public:
    const std::string&
    SnapshotName() const;

    void
    SetSnapshotName(std::string snapshot_name);

    const std::string&
    DatabaseName() const;

    void
    SetDatabaseName(std::string db_name);

    const std::string&
    CollectionName() const;

    void
    SetCollectionName(std::string collection_name);

    int64_t
    JobID() const;

    void
    SetJobID(int64_t job_id);

    RestoreSnapshotStateCode
    State() const;

    void
    SetState(RestoreSnapshotStateCode state);

    int32_t
    Progress() const;

    void
    SetProgress(int32_t progress);

    const std::string&
    Reason() const;

    void
    SetReason(std::string reason);

    uint64_t
    StartTime() const;

    void
    SetStartTime(uint64_t start_time);

    uint64_t
    TimeCost() const;

    void
    SetTimeCost(uint64_t time_cost);

 private:
    int64_t job_id_{0};
    std::string snapshot_name_;
    std::string db_name_;
    std::string collection_name_;
    RestoreSnapshotStateCode state_{RestoreSnapshotStateCode::UNKNOWN};
    int32_t progress_{0};
    std::string reason_;
    uint64_t start_time_{0};
    uint64_t time_cost_{0};
};

}  // namespace milvus
