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

#include "milvus/request/utility/OptimizeRequest.h"

namespace milvus {

const std::string&
OptimizeRequest::DatabaseName() const {
    return db_name_;
}

void
OptimizeRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

OptimizeRequest&
OptimizeRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::string&
OptimizeRequest::CollectionName() const {
    return collection_name_;
}

void
OptimizeRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

OptimizeRequest&
OptimizeRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

const std::string&
OptimizeRequest::TargetSize() const {
    return target_size_;
}

void
OptimizeRequest::SetTargetSize(const std::string& target_size) {
    target_size_ = target_size;
}

OptimizeRequest&
OptimizeRequest::WithTargetSize(const std::string& target_size) {
    SetTargetSize(target_size);
    return *this;
}

bool
OptimizeRequest::Async() const {
    return async_;
}

void
OptimizeRequest::SetAsync(bool async) {
    async_ = async;
}

OptimizeRequest&
OptimizeRequest::WithAsync(bool async) {
    SetAsync(async);
    return *this;
}

int64_t
OptimizeRequest::TimeoutMs() const {
    return timeout_ms_;
}

void
OptimizeRequest::SetTimeoutMs(int64_t timeout_ms) {
    timeout_ms_ = timeout_ms;
}

OptimizeRequest&
OptimizeRequest::WithTimeoutMs(int64_t timeout_ms) {
    SetTimeoutMs(timeout_ms);
    return *this;
}

}  // namespace milvus
