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

#include "milvus/request/index/CreateIndexRequest.h"

#include <memory>

namespace milvus {

CreateIndexRequest&
CreateIndexRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

CreateIndexRequest&
CreateIndexRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

const std::vector<IndexDesc>&
CreateIndexRequest::Indexes() const {
    return indexes_;
}

void
CreateIndexRequest::SetIndexes(std::vector<IndexDesc>&& indexes) {
    indexes_ = std::move(indexes);
}

CreateIndexRequest&
CreateIndexRequest::WithIndexes(std::vector<IndexDesc>&& indexes) {
    SetIndexes(std::move(indexes));
    return *this;
}

CreateIndexRequest&
CreateIndexRequest::AddIndex(IndexDesc&& index) {
    indexes_.emplace_back(std::move(index));
    return *this;
}

bool
CreateIndexRequest::Sync() const {
    return sync_;
}

void
CreateIndexRequest::SetSync(bool sync) {
    sync_ = sync;
}

CreateIndexRequest&
CreateIndexRequest::WithSync(bool sync) {
    SetSync(sync);
    return *this;
}

int64_t
CreateIndexRequest::TimeoutMs() const {
    return timeout_ms_;
}

void
CreateIndexRequest::SetTimeoutMs(int64_t timeout_ms) {
    timeout_ms_ = timeout_ms;
}

CreateIndexRequest&
CreateIndexRequest::WithTimeoutMs(int64_t timeout_ms) {
    SetTimeoutMs(timeout_ms);
    return *this;
}

}  // namespace milvus