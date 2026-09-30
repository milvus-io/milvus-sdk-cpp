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

int64_t
CompactionPlan::PlanId() const {
    return plan_id_;
}

void
CompactionPlan::SetPlanId(int64_t plan_id) {
    plan_id_ = plan_id;
}

int64_t
CompactionPlan::TriggerId() const {
    return trigger_id_;
}

void
CompactionPlan::SetTriggerId(int64_t trigger_id) {
    trigger_id_ = trigger_id;
}

int64_t
CompactionPlan::CollectionId() const {
    return collection_id_;
}

void
CompactionPlan::SetCollectionId(int64_t collection_id) {
    collection_id_ = collection_id;
}

int64_t
CompactionPlan::PartitionId() const {
    return partition_id_;
}

void
CompactionPlan::SetPartitionId(int64_t partition_id) {
    partition_id_ = partition_id;
}

const std::string&
CompactionPlan::Channel() const {
    return channel_;
}

void
CompactionPlan::SetChannel(const std::string& channel) {
    channel_ = channel;
}

CompactionType
CompactionPlan::Type() const {
    return type_;
}

void
CompactionPlan::SetType(CompactionType type) {
    type_ = type;
}

CompactionTaskState
CompactionPlan::State() const {
    return state_;
}

void
CompactionPlan::SetState(CompactionTaskState state) {
    state_ = state;
}

const std::string&
CompactionPlan::FailureReason() const {
    return failure_reason_;
}

void
CompactionPlan::SetFailureReason(const std::string& failure_reason) {
    failure_reason_ = failure_reason;
}

const std::vector<int64_t>&
CompactionPlan::Targets() const {
    return targets_;
}

void
CompactionPlan::SetTargets(const std::vector<int64_t>& targets) {
    targets_ = targets;
}

void
CompactionPlan::SetTargets(std::vector<int64_t>&& targets) {
    targets_ = std::move(targets);
}

}  // namespace milvus
