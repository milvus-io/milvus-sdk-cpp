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

#include "milvus/request/utility/ListRunningRequestsRequest.h"

namespace milvus {

const std::string&
ListRunningRequestsRequest::DatabaseName() const {
    return db_name_;
}

void
ListRunningRequestsRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

ListRunningRequestsRequest&
ListRunningRequestsRequest::WithDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
    return *this;
}

const std::string&
ListRunningRequestsRequest::CollectionName() const {
    return collection_name_;
}

void
ListRunningRequestsRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

ListRunningRequestsRequest&
ListRunningRequestsRequest::WithCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
    return *this;
}

const std::string&
ListRunningRequestsRequest::User() const {
    return user_;
}

void
ListRunningRequestsRequest::SetUser(const std::string& user) {
    user_ = user;
}

ListRunningRequestsRequest&
ListRunningRequestsRequest::WithUser(const std::string& user) {
    user_ = user;
    return *this;
}

int64_t
ListRunningRequestsRequest::MinElapsedMs() const {
    return min_elapsed_ms_;
}

void
ListRunningRequestsRequest::SetMinElapsedMs(int64_t min_elapsed_ms) {
    min_elapsed_ms_ = min_elapsed_ms;
}

ListRunningRequestsRequest&
ListRunningRequestsRequest::WithMinElapsedMs(int64_t min_elapsed_ms) {
    min_elapsed_ms_ = min_elapsed_ms;
    return *this;
}

}  // namespace milvus
