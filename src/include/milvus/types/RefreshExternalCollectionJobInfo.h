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
#include "milvus/types/RefreshExternalCollectionState.h"

namespace milvus {

/**
 * @brief Refresh external collection job information.
 */
class MILVUS_SDK_API RefreshExternalCollectionJobInfo {
 public:
    int64_t
    JobID() const;

    void
    SetJobID(int64_t job_id);

    const std::string&
    CollectionName() const;

    void
    SetCollectionName(std::string collection_name);

    RefreshExternalCollectionStateCode
    State() const;

    void
    SetState(RefreshExternalCollectionStateCode state);

    int32_t
    Progress() const;

    void
    SetProgress(int32_t progress);

    const std::string&
    Reason() const;

    void
    SetReason(std::string reason);

    const std::string&
    ExternalSource() const;

    void
    SetExternalSource(std::string external_source);

    uint64_t
    StartTime() const;

    void
    SetStartTime(uint64_t start_time);

    uint64_t
    EndTime() const;

    void
    SetEndTime(uint64_t end_time);

 private:
    int64_t job_id_{0};
    std::string collection_name_;
    RefreshExternalCollectionStateCode state_{RefreshExternalCollectionStateCode::PENDING};
    int32_t progress_{0};
    std::string reason_;
    std::string external_source_;
    uint64_t start_time_{0};
    uint64_t end_time_{0};
};

}  // namespace milvus
