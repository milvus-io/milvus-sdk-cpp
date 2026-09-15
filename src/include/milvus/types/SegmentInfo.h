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
#include <vector>

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief State of segment
 */
enum class SegmentState {
    /**
     * @brief Segment state is unknown.
     */
    UNKNOWN = 0,
    /**
     * @brief The segment does not exist.
     */
    NOT_EXIST = 1,
    /**
     * @brief The segment is growing and accepting writes.
     */
    GROWING = 2,
    /**
     * @brief The segment is sealed and no longer accepts writes.
     */
    SEALED = 3,
    /**
     * @brief The segment has been flushed to storage.
     */
    FLUSHED = 4,
    /**
     * @brief The segment is being flushed to storage.
     */
    FLUSHING = 5,
    /**
     * @brief The segment has been dropped.
     */
    DROPPED = 6,
};

/**
 * @brief Level of segment.
 */
enum class SegmentLevel {
    /**
     * @brief Segment state is unknown.
     */
    UNKNOWN = -1,
    /**
     * @brief Legacy segment level.
     */
    LEGACY = 0,
    /**
     * @brief Level-0 segment (delta log).
     */
    L0 = 1,
    /**
     * @brief Level-1 sealed segment.
     */
    L1 = 2,
    /**
     * @brief Level-2 sealed segment.
     */
    L2 = 3,
};

/**
 * @brief Persisted segment information returned by MilvusClient::GetPersistentSegmentInfo().
 */
class MILVUS_SDK_API SegmentInfo {
 public:
    /**
     * @brief Constructor
     * @param [in] collection_id the collection ID.
     * @param [in] partition_id the partition ID.
     * @param [in] segment_id the segment ID.
     * @param [in] row_count the row count.
     * @param [in] state the state.
     */
    SegmentInfo(int64_t collection_id, int64_t partition_id, int64_t segment_id, int64_t row_count, SegmentState state);

    /**
     * @brief Constructor
     */
    SegmentInfo(int64_t collection_id, int64_t partition_id, int64_t segment_id, int64_t row_count, SegmentState state,
                std::string collection_name, SegmentLevel level, int64_t storage_version, bool is_sorted);

    /**
     * @brief The collection id which this segment belong to.
     * @return the collection ID.
     */
    int64_t
    CollectionID() const;

    /**
     * @brief The partition id which this segment belong to.
     * @return the partition ID.
     */
    int64_t
    PartitionID() const;

    /**
     * @brief ID of the segment.
     * @return the segment ID.
     */
    int64_t
    SegmentID() const;
    /**
     * @brief Row count of the segment.
     * @return the row count.
     */
    int64_t
    RowCount() const;

    /**
     * @brief Current state of the segment.
     * @return the state.
     */
    SegmentState
    State() const;

    /**
     * @brief The collection name which this segment belongs to.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Level of the segment.
     * @return the level.
     */
    SegmentLevel
    Level() const;

    /**
     * @brief Storage version of the segment.
     * @return the storage version.
     */
    int64_t
    StorageVersion() const;

    /**
     * @brief Whether the segment is sorted.
     * @return true if the results are sorted.
     */
    bool
    IsSorted() const;

 private:
    int64_t collection_id_{0};
    int64_t partition_id_{0};
    int64_t segment_id_{0};
    int64_t row_count_{0};

    SegmentState state_{SegmentState::UNKNOWN};
    std::string collection_name_;
    SegmentLevel level_{SegmentLevel::UNKNOWN};
    int64_t storage_version_{0};
    bool is_sorted_{false};
};

/**
 * @brief SegmentsInfo objects array
 */
using SegmentsInfo = std::vector<SegmentInfo>;

/**
 * @brief In-memory segment information returned by MilvusClient::GetQuerySegmentInfo().
 */
class MILVUS_SDK_API QuerySegmentInfo : public SegmentInfo {
 public:
    /**
     * @brief Constructor
     */
    QuerySegmentInfo(int64_t collection_id, int64_t partition_id, int64_t segment_id, int64_t row_count,
                     SegmentState state, std::string index_name, int64_t index_id,
                     const std::vector<int64_t>& node_ids);

    /**
     * @brief Constructor
     */
    QuerySegmentInfo(int64_t collection_id, int64_t partition_id, int64_t segment_id, int64_t row_count,
                     SegmentState state, std::string index_name, int64_t index_id, const std::vector<int64_t>& node_ids,
                     std::string collection_name, int64_t mem_size, SegmentLevel level, int64_t storage_version,
                     bool is_sorted);

    /**
     * @brief Index name of the segment.
     * @return the index name.
     */
    std::string
    IndexName() const;

    /**
     * @brief Index id the segment.
     * @return the index ID.
     */
    int64_t
    IndexID() const;

    /**
     * @brief Node id of the segment.
     * @deprecated in v2.4, a segment can be loaded into multiple nodes, use the NodeIDs() instead.
     * This method will return the first node id in from id list.
     * @return the node ID.
     */
    int64_t
    NodeID() const;

    /**
     * @brief Node id list of the segment.
     * @return the node i ds.
     */
    const std::vector<int64_t>&
    NodeIDs() const;

    /**
     * @brief Memory size of the segment.
     * @return the mem size.
     */
    int64_t
    MemSize() const;

 private:
    std::string index_name_;
    int64_t index_id_{0};
    std::vector<int64_t> node_ids_;
    int64_t mem_size_{0};
};

/**
 * @brief QuerySegmentsInfo objects array
 */
using QuerySegmentsInfo = std::vector<QuerySegmentInfo>;

}  // namespace milvus
