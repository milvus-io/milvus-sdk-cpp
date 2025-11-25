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

#include "milvus/response/utility/CompactResponse.h"

namespace milvus {

int64_t
CompactResponse::CompactionID() const {
    return compaction_id_;
}

void
CompactResponse::SetCompactionID(int64_t id) {
    compaction_id_ = id;
}

int64_t
CompactResponse::CompactionPlanCount() const {
    return compaction_plan_count_;
}

void
CompactResponse::SetCompactionPlanCount(int64_t count) {
    compaction_plan_count_ = count;
}

}  // namespace milvus
