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

#include "milvus/types/RefreshExternalCollectionJobInfo.h"

namespace milvus {

int64_t
RefreshExternalCollectionJobInfo::JobID() const {
    return job_id_;
}

void
RefreshExternalCollectionJobInfo::SetJobID(int64_t job_id) {
    job_id_ = job_id;
}

const std::string&
RefreshExternalCollectionJobInfo::CollectionName() const {
    return collection_name_;
}

void
RefreshExternalCollectionJobInfo::SetCollectionName(std::string collection_name) {
    collection_name_ = std::move(collection_name);
}

RefreshExternalCollectionStateCode
RefreshExternalCollectionJobInfo::State() const {
    return state_;
}

void
RefreshExternalCollectionJobInfo::SetState(RefreshExternalCollectionStateCode state) {
    state_ = state;
}

int32_t
RefreshExternalCollectionJobInfo::Progress() const {
    return progress_;
}

void
RefreshExternalCollectionJobInfo::SetProgress(int32_t progress) {
    progress_ = progress;
}

const std::string&
RefreshExternalCollectionJobInfo::Reason() const {
    return reason_;
}

void
RefreshExternalCollectionJobInfo::SetReason(std::string reason) {
    reason_ = std::move(reason);
}

const std::string&
RefreshExternalCollectionJobInfo::ExternalSource() const {
    return external_source_;
}

void
RefreshExternalCollectionJobInfo::SetExternalSource(std::string external_source) {
    external_source_ = std::move(external_source);
}

uint64_t
RefreshExternalCollectionJobInfo::StartTime() const {
    return start_time_;
}

void
RefreshExternalCollectionJobInfo::SetStartTime(uint64_t start_time) {
    start_time_ = start_time;
}

uint64_t
RefreshExternalCollectionJobInfo::EndTime() const {
    return end_time_;
}

void
RefreshExternalCollectionJobInfo::SetEndTime(uint64_t end_time) {
    end_time_ = end_time;
}

}  // namespace milvus
