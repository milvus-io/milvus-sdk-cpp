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
#include <vector>

namespace milvus {

/**
 * @brief State of segment
 */
enum class SegmentState {
    UNKNOWN = 0,
    NOT_EXIST = 1,
    GROWING = 2,
    SEALED = 3,
    FLUSHED = 4,
    FLUSHING = 5,
    DROPPED = 6,
};

/**
 * @brief Persisted segment information returned by GetPersistentSegmentInfo().
 */
class SegmentInfo {
 public:
    SegmentInfo(int64_t collection_id, int64_t partition_id, int64_t segment_id, int64_t row_count, SegmentState state)
        : collection_id_{collection_id},
          partition_id_{partition_id},
          segment_id_{segment_id},
          row_count_{row_count},
          state_(state) {
    }

    /**
     * @brief The collection id which this segment belong to.
     */
    int64_t
    CollectionID() const {
        return collection_id_;
    }

    /**
     * @brief The partition id which this segment belong to.
     */
    int64_t
    PartitionID() const {
        return partition_id_;
    }

    /**
     * @brief ID of the segment.
     */
    int64_t
    SegmentID() const {
        return segment_id_;
    }

    /**
     * @brief Row count of the segment.
     */
    int64_t
    RowCount() const {
        return row_count_;
    }

    /**
     * @brief Current state of the segment.
     */
    SegmentState
    State() const {
        return state_;
    }

 private:
    int64_t collection_id_ = 0;
    int64_t partition_id_ = 0;
    int64_t segment_id_ = 0;
    int64_t row_count_ = 0;

    SegmentState state_{SegmentState::UNKNOWN};
};

using SegmentsInfo = std::vector<SegmentInfo>;

class QuerySegmentInfo : public SegmentInfo {
 public:
    QuerySegmentInfo(int64_t collection_id, int64_t partition_id, int64_t segment_id, int64_t row_count,
                     SegmentState state, const std::string& index_name, int64_t index_id, int64_t node_id)
        : SegmentInfo(collection_id, partition_id, segment_id, row_count, state),
          index_name_{index_name},
          index_id_{index_id},
          node_id_{node_id} {
    }

    /**
     * @brief Index name of the segment.
     */
    std::string
    IndexName() const {
        return index_name_;
    }

    /**
     * @brief Index id the segment.
     */
    int64_t
    IndexID() const {
        return index_id_;
    }

    /**
     * @brief Node id of the segment.
     */
    int64_t
    NodeID() const {
        return node_id_;
    }

 private:
    std::string index_name_;
    int64_t index_id_ = 0;
    int64_t node_id_ = 0;
};

using QuerySegmentsInfo = std::vector<QuerySegmentInfo>;

}  // namespace milvus
