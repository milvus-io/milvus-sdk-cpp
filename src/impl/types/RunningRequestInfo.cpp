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

#include "milvus/types/RunningRequestInfo.h"

#include <utility>

namespace milvus {

const std::string&
RunningRequestInfo::RequestId() const {
    return request_id_;
}

void
RunningRequestInfo::SetRequestId(const std::string& request_id) {
    request_id_ = request_id;
}

RunningRequestInfo&
RunningRequestInfo::WithRequestId(const std::string& request_id) {
    request_id_ = request_id;
    return *this;
}

int64_t
RunningRequestInfo::ProxyId() const {
    return proxy_id_;
}

void
RunningRequestInfo::SetProxyId(int64_t proxy_id) {
    proxy_id_ = proxy_id;
}

RunningRequestInfo&
RunningRequestInfo::WithProxyId(int64_t proxy_id) {
    proxy_id_ = proxy_id;
    return *this;
}

const std::string&
RunningRequestInfo::Type() const {
    return type_;
}

void
RunningRequestInfo::SetType(const std::string& type) {
    type_ = type;
}

RunningRequestInfo&
RunningRequestInfo::WithType(const std::string& type) {
    type_ = type;
    return *this;
}

const std::string&
RunningRequestInfo::DatabaseName() const {
    return db_name_;
}

void
RunningRequestInfo::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

RunningRequestInfo&
RunningRequestInfo::WithDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
    return *this;
}

const std::string&
RunningRequestInfo::CollectionName() const {
    return collection_name_;
}

void
RunningRequestInfo::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

RunningRequestInfo&
RunningRequestInfo::WithCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
    return *this;
}

const std::string&
RunningRequestInfo::User() const {
    return user_;
}

void
RunningRequestInfo::SetUser(const std::string& user) {
    user_ = user;
}

RunningRequestInfo&
RunningRequestInfo::WithUser(const std::string& user) {
    user_ = user;
    return *this;
}

const std::string&
RunningRequestInfo::ClientAddr() const {
    return client_addr_;
}

void
RunningRequestInfo::SetClientAddr(const std::string& client_addr) {
    client_addr_ = client_addr;
}

RunningRequestInfo&
RunningRequestInfo::WithClientAddr(const std::string& client_addr) {
    client_addr_ = client_addr;
    return *this;
}

int64_t
RunningRequestInfo::Nq() const {
    return nq_;
}

void
RunningRequestInfo::SetNq(int64_t nq) {
    nq_ = nq;
}

RunningRequestInfo&
RunningRequestInfo::WithNq(int64_t nq) {
    nq_ = nq;
    return *this;
}

int64_t
RunningRequestInfo::TopK() const {
    return top_k_;
}

void
RunningRequestInfo::SetTopK(int64_t top_k) {
    top_k_ = top_k;
}

RunningRequestInfo&
RunningRequestInfo::WithTopK(int64_t top_k) {
    top_k_ = top_k;
    return *this;
}

const std::string&
RunningRequestInfo::Expr() const {
    return expr_;
}

void
RunningRequestInfo::SetExpr(const std::string& expr) {
    expr_ = expr;
}

RunningRequestInfo&
RunningRequestInfo::WithExpr(const std::string& expr) {
    expr_ = expr;
    return *this;
}

int64_t
RunningRequestInfo::StartTimeMs() const {
    return start_time_ms_;
}

void
RunningRequestInfo::SetStartTimeMs(int64_t start_time_ms) {
    start_time_ms_ = start_time_ms;
}

RunningRequestInfo&
RunningRequestInfo::WithStartTimeMs(int64_t start_time_ms) {
    start_time_ms_ = start_time_ms;
    return *this;
}

int64_t
RunningRequestInfo::QueuedMs() const {
    return queued_ms_;
}

void
RunningRequestInfo::SetQueuedMs(int64_t queued_ms) {
    queued_ms_ = queued_ms;
}

RunningRequestInfo&
RunningRequestInfo::WithQueuedMs(int64_t queued_ms) {
    queued_ms_ = queued_ms;
    return *this;
}

int64_t
RunningRequestInfo::ElapsedMs() const {
    return elapsed_ms_;
}

void
RunningRequestInfo::SetElapsedMs(int64_t elapsed_ms) {
    elapsed_ms_ = elapsed_ms;
}

RunningRequestInfo&
RunningRequestInfo::WithElapsedMs(int64_t elapsed_ms) {
    elapsed_ms_ = elapsed_ms;
    return *this;
}

const std::string&
RunningRequestInfo::State() const {
    return state_;
}

void
RunningRequestInfo::SetState(const std::string& state) {
    state_ = state;
}

RunningRequestInfo&
RunningRequestInfo::WithState(const std::string& state) {
    state_ = state;
    return *this;
}

const std::vector<int64_t>&
RunningRequestInfo::TaskIds() const {
    return task_ids_;
}

void
RunningRequestInfo::SetTaskIds(const std::vector<int64_t>& task_ids) {
    task_ids_ = task_ids;
}

void
RunningRequestInfo::SetTaskIds(std::vector<int64_t>&& task_ids) {
    task_ids_ = std::move(task_ids);
}

RunningRequestInfo&
RunningRequestInfo::WithTaskIds(std::vector<int64_t>&& task_ids) {
    task_ids_ = std::move(task_ids);
    return *this;
}

const std::string&
RunningRequestInfo::TraceId() const {
    return trace_id_;
}

void
RunningRequestInfo::SetTraceId(const std::string& trace_id) {
    trace_id_ = trace_id;
}

RunningRequestInfo&
RunningRequestInfo::WithTraceId(const std::string& trace_id) {
    trace_id_ = trace_id;
    return *this;
}

bool
RunningRequestInfo::Cancellable() const {
    return cancellable_;
}

void
RunningRequestInfo::SetCancellable(bool cancellable) {
    cancellable_ = cancellable;
}

RunningRequestInfo&
RunningRequestInfo::WithCancellable(bool cancellable) {
    cancellable_ = cancellable;
    return *this;
}

}  // namespace milvus
