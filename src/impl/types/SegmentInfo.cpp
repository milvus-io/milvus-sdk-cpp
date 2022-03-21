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

#include "milvus/types/SegmentInfo.h"

namespace milvus {

SegmentInfo::SegmentInfo(int64_t collection_id, int64_t partition_id, int64_t segment_id, int64_t row_count,
                         SegmentState state)
    : collection_id_{collection_id},
      partition_id_{partition_id},
      segment_id_{segment_id},
      row_count_{row_count},
      state_(state) {
}

int64_t
SegmentInfo::CollectionID() const {
    return collection_id_;
}

int64_t
SegmentInfo::PartitionID() const {
    return partition_id_;
}

int64_t
SegmentInfo::SegmentID() const {
    return segment_id_;
}

int64_t
SegmentInfo::RowCount() const {
    return row_count_;
}

SegmentState
SegmentInfo::State() const {
    return state_;
}

QuerySegmentInfo::QuerySegmentInfo(int64_t collection_id, int64_t partition_id, int64_t segment_id, int64_t row_count,
                                   SegmentState state, std::string index_name, int64_t index_id, int64_t node_id)
    : SegmentInfo(collection_id, partition_id, segment_id, row_count, state),
      index_name_{std::move(index_name)},
      index_id_{index_id},
      node_id_{node_id} {
}

std::string
QuerySegmentInfo::IndexName() const {
    return index_name_;
}

int64_t
QuerySegmentInfo::IndexID() const {
    return index_id_;
}

int64_t
QuerySegmentInfo::NodeID() const {
    return node_id_;
}

}  // namespace milvus
