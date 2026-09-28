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
 * @brief Type of one compaction task. Numeric values mirror common.proto CompactionType.
 */
enum class CompactionType {
    UNDEFINED = 0,
    MERGE = 2,
    MIX = 3,
    SINGLE = 4,
    MINOR = 5,
    MAJOR = 6,
    LEVEL0_DELETE = 7,
    CLUSTERING = 8,
    SORT = 9,
    PARTITION_KEY_SORT = 10,
    CLUSTERING_PARTITION_KEY_SORT = 11,
    BUMP_SCHEMA_VERSION = 12,
};

/**
 * @brief State of one compaction task. Numeric values mirror common.proto CompactionTaskState.
 */
enum class CompactionTaskState {
    UNKNOWN = 0,
    EXECUTING = 1,
    PIPELINING = 2,
    COMPLETED = 3,
    FAILED = 4,
    TIMEOUT = 5,
    ANALYZING = 6,
    INDEXING = 7,
    CLEANED = 8,
    META_SAVED = 9,
    STATISTIC = 10,
};

/**
 * @brief Compaction plan information. Used by MilvusClient::GetCompactionPlans() and
 * MilvusClientV2::ListCompactionTasks().
 */
class MILVUS_SDK_API CompactionPlan {
 public:
    /**
     * @brief Construct a new Compaction Plan object.
     */
    CompactionPlan();

    /**
     * @brief Constructor
     * @param [in] segments the segments.
     * @param [in] dst_segment the dst segment.
     */
    CompactionPlan(const std::vector<int64_t>& segments, int64_t dst_segment);

    /**
     * @brief Constructor
     * @param [in] segments the segments.
     * @param [in] dst_segment the dst segment.
     */
    CompactionPlan(std::vector<int64_t>&& segments, int64_t dst_segment);

    /**
     * @brief Segment id array to be merged.
     * @return the source segments.
     */
    const std::vector<int64_t>&
    SourceSegments() const;

    /**
     * @brief Set segment id array to be merged.
     * @param [in] segments the segments.
     */
    void
    SetSourceSegments(const std::vector<int64_t>& segments);

    /**
     * @brief Set segment id array to be merged.
     * @param [in] segments the segments.
     */
    void
    SetSourceSegments(std::vector<int64_t>&& segments);

    /**
     * @brief New generated segment id after merging.
     * @return the destiny segemnt.
     */
    int64_t
    DestinySegemnt() const;

    /**
     * @brief Set segment id.
     * @param [in] id the ID.
     */
    void
    SetDestinySegemnt(int64_t id);

    /**
     * @brief The server-side compaction task identifier.
     * @return the plan id.
     */
    int64_t
    PlanId() const;

    /**
     * @brief Set the server-side compaction task identifier.
     * @param [in] plan_id the plan id.
     */
    void
    SetPlanId(int64_t plan_id);

    /**
     * @brief The compaction trigger id.
     * @return the trigger id.
     */
    int64_t
    TriggerId() const;

    /**
     * @brief Set the compaction trigger id.
     * @param [in] trigger_id the trigger id.
     */
    void
    SetTriggerId(int64_t trigger_id);

    /**
     * @brief The collection id of this compaction task.
     * @return the collection id.
     */
    int64_t
    CollectionId() const;

    /**
     * @brief Set the collection id of this compaction task.
     * @param [in] collection_id the collection id.
     */
    void
    SetCollectionId(int64_t collection_id);

    /**
     * @brief The partition id of this compaction task.
     * @return the partition id.
     */
    int64_t
    PartitionId() const;

    /**
     * @brief Set the partition id of this compaction task.
     * @param [in] partition_id the partition id.
     */
    void
    SetPartitionId(int64_t partition_id);

    /**
     * @brief The channel of this compaction task.
     * @return the channel.
     */
    const std::string&
    Channel() const;

    /**
     * @brief Set the channel of this compaction task.
     * @param [in] channel the channel.
     */
    void
    SetChannel(const std::string& channel);

    /**
     * @brief The type of this compaction task.
     * @return the compaction type.
     */
    CompactionType
    Type() const;

    /**
     * @brief Set the type of this compaction task.
     * @param [in] type the compaction type.
     */
    void
    SetType(CompactionType type);

    /**
     * @brief The state of this compaction task.
     * @return the compaction task state.
     */
    CompactionTaskState
    State() const;

    /**
     * @brief Set the state of this compaction task.
     * @param [in] state the compaction task state.
     */
    void
    SetState(CompactionTaskState state);

    /**
     * @brief The failure reason of this compaction task, empty when it succeeded.
     * @return the failure reason.
     */
    const std::string&
    FailureReason() const;

    /**
     * @brief Set the failure reason of this compaction task.
     * @param [in] failure_reason the failure reason.
     */
    void
    SetFailureReason(const std::string& failure_reason);

    /**
     * @brief The complete output segment set. Prefer over DestinySegemnt().
     * @return the targets.
     */
    const std::vector<int64_t>&
    Targets() const;

    /**
     * @brief Set the complete output segment set.
     * @param [in] targets the targets.
     */
    void
    SetTargets(std::vector<int64_t>&& targets);

 private:
    std::vector<int64_t> src_segments_;
    int64_t dst_segment_ = 0;
    int64_t plan_id_ = 0;
    int64_t trigger_id_ = 0;
    int64_t collection_id_ = 0;
    int64_t partition_id_ = 0;
    std::string channel_;
    CompactionType type_{CompactionType::UNDEFINED};
    CompactionTaskState state_{CompactionTaskState::UNKNOWN};
    std::string failure_reason_;
    std::vector<int64_t> targets_;
};

/**
 * @brief CompactionPlans objects array
 */
using CompactionPlans = std::vector<CompactionPlan>;

}  // namespace milvus
