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

#include "milvus/request/utility/FlushRequest.h"

#include <memory>

namespace milvus {

const std::string&
FlushRequest::DatabaseName() const {
    return db_name_;
}

void
FlushRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

FlushRequest&
FlushRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::set<std::string>&
FlushRequest::CollectionNames() const {
    return collection_names_;
}

void
FlushRequest::SetCollectionNames(std::set<std::string>&& names) {
    collection_names_ = std::move(names);
}

FlushRequest&
FlushRequest::WithCollectionNames(std::set<std::string>&& names) {
    SetCollectionNames(std::move(names));
    return *this;
}

FlushRequest&
FlushRequest::AddCollectionName(const std::string& name) {
    collection_names_.insert(name);
    return *this;
}

int64_t
FlushRequest::WaitFlushedMs() const {
    return wait_flushed_ms_;
}

void
FlushRequest::SetWaitFlushedMs(int64_t ms) {
    wait_flushed_ms_ = ms;
}

FlushRequest&
FlushRequest::WithWaitFlushedMs(int64_t ms) {
    SetWaitFlushedMs(ms);
    return *this;
}

}  // namespace milvus
