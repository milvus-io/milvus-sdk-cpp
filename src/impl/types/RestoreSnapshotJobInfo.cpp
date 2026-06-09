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

#include "milvus/types/RestoreSnapshotJobInfo.h"

namespace milvus {

const std::string&
RestoreSnapshotJobInfo::SnapshotName() const {
    return snapshot_name_;
}

void
RestoreSnapshotJobInfo::SetSnapshotName(std::string snapshot_name) {
    snapshot_name_ = std::move(snapshot_name);
}

const std::string&
RestoreSnapshotJobInfo::DatabaseName() const {
    return db_name_;
}

void
RestoreSnapshotJobInfo::SetDatabaseName(std::string db_name) {
    db_name_ = std::move(db_name);
}

const std::string&
RestoreSnapshotJobInfo::CollectionName() const {
    return collection_name_;
}

void
RestoreSnapshotJobInfo::SetCollectionName(std::string collection_name) {
    collection_name_ = std::move(collection_name);
}

int64_t
RestoreSnapshotJobInfo::JobID() const {
    return job_id_;
}

void
RestoreSnapshotJobInfo::SetJobID(int64_t job_id) {
    job_id_ = job_id;
}

RestoreSnapshotStateCode
RestoreSnapshotJobInfo::State() const {
    return state_;
}

void
RestoreSnapshotJobInfo::SetState(RestoreSnapshotStateCode state) {
    state_ = state;
}

int32_t
RestoreSnapshotJobInfo::Progress() const {
    return progress_;
}

void
RestoreSnapshotJobInfo::SetProgress(int32_t progress) {
    progress_ = progress;
}

const std::string&
RestoreSnapshotJobInfo::Reason() const {
    return reason_;
}

void
RestoreSnapshotJobInfo::SetReason(std::string reason) {
    reason_ = std::move(reason);
}

uint64_t
RestoreSnapshotJobInfo::StartTime() const {
    return start_time_;
}

void
RestoreSnapshotJobInfo::SetStartTime(uint64_t start_time) {
    start_time_ = start_time;
}

uint64_t
RestoreSnapshotJobInfo::TimeCost() const {
    return time_cost_;
}

void
RestoreSnapshotJobInfo::SetTimeCost(uint64_t time_cost) {
    time_cost_ = time_cost;
}

}  // namespace milvus
