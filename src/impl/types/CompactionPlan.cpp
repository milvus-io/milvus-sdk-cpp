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

#include "milvus/types/CompactionPlan.h"

namespace milvus {

CompactionPlan::CompactionPlan() = default;

CompactionPlan::CompactionPlan(const std::vector<int64_t>& segments, int64_t dst_segment)
    : src_segments_(segments), dst_segment_(dst_segment) {
}

CompactionPlan::CompactionPlan(std::vector<int64_t>&& segments, int64_t dst_segment)
    : src_segments_(std::move(segments)), dst_segment_(dst_segment) {
}

const std::vector<int64_t>&
CompactionPlan::SourceSegments() const {
    return src_segments_;
}

void
CompactionPlan::SetSourceSegments(const std::vector<int64_t>& segments) {
    src_segments_ = segments;
}

void
CompactionPlan::SetSourceSegments(std::vector<int64_t>&& segments) {
    src_segments_ = std::move(segments);
}

int64_t
CompactionPlan::DestinySegemnt() const {
    return dst_segment_;
}

void
CompactionPlan::SetDestinySegemnt(int64_t id) {
    dst_segment_ = id;
}

}  // namespace milvus
