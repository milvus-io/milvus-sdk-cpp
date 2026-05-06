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

#include "milvus/response/utility/OptimizeResponse.h"

namespace milvus {

const std::string&
OptimizeResponse::StatusText() const {
    return status_;
}

void
OptimizeResponse::SetStatusText(const std::string& status) {
    status_ = status;
}

const std::string&
OptimizeResponse::CollectionName() const {
    return collection_name_;
}

void
OptimizeResponse::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

int64_t
OptimizeResponse::CompactionID() const {
    return compaction_id_;
}

void
OptimizeResponse::SetCompactionID(int64_t compaction_id) {
    compaction_id_ = compaction_id;
}

const std::string&
OptimizeResponse::TargetSize() const {
    return target_size_;
}

void
OptimizeResponse::SetTargetSize(const std::string& target_size) {
    target_size_ = target_size;
}

const std::vector<std::string>&
OptimizeResponse::ProgressHistory() const {
    return progress_history_;
}

void
OptimizeResponse::SetProgressHistory(std::vector<std::string>&& progress_history) {
    progress_history_ = std::move(progress_history);
}

void
OptimizeResponse::AddProgress(const std::string& progress) {
    progress_history_.push_back(progress);
}

}  // namespace milvus
