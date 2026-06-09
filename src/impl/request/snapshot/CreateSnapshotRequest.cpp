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

#include "milvus/request/snapshot/CreateSnapshotRequest.h"

namespace milvus {

const std::string&
CreateSnapshotRequest::Description() const {
    return description_;
}

void
CreateSnapshotRequest::SetDescription(const std::string& description) {
    description_ = description;
}

CreateSnapshotRequest&
CreateSnapshotRequest::WithDescription(const std::string& description) {
    SetDescription(description);
    return *this;
}

int64_t
CreateSnapshotRequest::CompactionProtectionSeconds() const {
    return compaction_protection_seconds_;
}

void
CreateSnapshotRequest::SetCompactionProtectionSeconds(int64_t seconds) {
    compaction_protection_seconds_ = seconds;
}

CreateSnapshotRequest&
CreateSnapshotRequest::WithCompactionProtectionSeconds(int64_t seconds) {
    SetCompactionProtectionSeconds(seconds);
    return *this;
}

}  // namespace milvus
