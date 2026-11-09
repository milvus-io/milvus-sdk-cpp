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
 * @brief One Search, HybridSearch or Query request that a proxy is executing.
 * Used by MilvusClientV2::ListRunningRequests() and MilvusClientV2::CancelRequests().
 */
class MILVUS_SDK_API RunningRequestInfo {
 public:
    /**
     * @brief Constructor
     */
    RunningRequestInfo() = default;

    /**
     * @brief The request id, an opaque string such as "25-to03zu-rs"; pass it to
     * CancelRequests() unchanged.
     * @return the request id.
     */
    const std::string&
    RequestId() const;

    /**
     * @brief Set the request id.
     * @param [in] request_id the request id.
     */
    void
    SetRequestId(const std::string& request_id);

    /**
     * @brief Set the request id.
     * @param [in] request_id the request id.
     */
    RunningRequestInfo&
    WithRequestId(const std::string& request_id);

    /**
     * @brief The node id of the proxy that holds the request.
     * @return the proxy id.
     */
    int64_t
    ProxyId() const;

    /**
     * @brief Set the node id of the proxy that holds the request.
     * @param [in] proxy_id the proxy id.
     */
    void
    SetProxyId(int64_t proxy_id);

    /**
     * @brief Set the node id of the proxy that holds the request.
     * @param [in] proxy_id the proxy id.
     */
    RunningRequestInfo&
    WithProxyId(int64_t proxy_id);

    /**
     * @brief The request type: Search, HybridSearch or Query.
     * @return the type.
     */
    const std::string&
    Type() const;

    /**
     * @brief Set the request type.
     * @param [in] type the type.
     */
    void
    SetType(const std::string& type);

    /**
     * @brief Set the request type.
     * @param [in] type the type.
     */
    RunningRequestInfo&
    WithType(const std::string& type);

    /**
     * @brief The database name.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set the database name.
     * @param [in] db_name the database name.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set the database name.
     * @param [in] db_name the database name.
     */
    RunningRequestInfo&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief The collection name.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set the collection name.
     * @param [in] collection_name the collection name.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set the collection name.
     * @param [in] collection_name the collection name.
     */
    RunningRequestInfo&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief The user who issued the request.
     * @return the user.
     */
    const std::string&
    User() const;

    /**
     * @brief Set the user who issued the request.
     * @param [in] user the user.
     */
    void
    SetUser(const std::string& user);

    /**
     * @brief Set the user who issued the request.
     * @param [in] user the user.
     */
    RunningRequestInfo&
    WithUser(const std::string& user);

    /**
     * @brief The client address seen by the proxy.
     * @return the client address.
     */
    const std::string&
    ClientAddr() const;

    /**
     * @brief Set the client address seen by the proxy.
     * @param [in] client_addr the client address.
     */
    void
    SetClientAddr(const std::string& client_addr);

    /**
     * @brief Set the client address seen by the proxy.
     * @param [in] client_addr the client address.
     */
    RunningRequestInfo&
    WithClientAddr(const std::string& client_addr);

    /**
     * @brief The number of query vectors.
     * @return the nq.
     */
    int64_t
    Nq() const;

    /**
     * @brief Set the number of query vectors.
     * @param [in] nq the nq.
     */
    void
    SetNq(int64_t nq);

    /**
     * @brief Set the number of query vectors.
     * @param [in] nq the nq.
     */
    RunningRequestInfo&
    WithNq(int64_t nq);

    /**
     * @brief The top-k of the request.
     * @return the top-k.
     */
    int64_t
    TopK() const;

    /**
     * @brief Set the top-k of the request.
     * @param [in] top_k the top-k.
     */
    void
    SetTopK(int64_t top_k);

    /**
     * @brief Set the top-k of the request.
     * @param [in] top_k the top-k.
     */
    RunningRequestInfo&
    WithTopK(int64_t top_k);

    /**
     * @brief The leading part (at most 256 bytes) of the filter expression.
     * @return the expression.
     */
    const std::string&
    Expr() const;

    /**
     * @brief Set the filter expression.
     * @param [in] expr the expression.
     */
    void
    SetExpr(const std::string& expr);

    /**
     * @brief Set the filter expression.
     * @param [in] expr the expression.
     */
    RunningRequestInfo&
    WithExpr(const std::string& expr);

    /**
     * @brief The time the request started, in milliseconds since the Unix epoch.
     * @return the start time.
     */
    int64_t
    StartTimeMs() const;

    /**
     * @brief Set the time the request started.
     * @param [in] start_time_ms the start time.
     */
    void
    SetStartTimeMs(int64_t start_time_ms);

    /**
     * @brief Set the time the request started.
     * @param [in] start_time_ms the start time.
     */
    RunningRequestInfo&
    WithStartTimeMs(int64_t start_time_ms);

    /**
     * @brief How long the request's first task waited in the proxy queue, in milliseconds;
     * 0 until that task starts.
     * @return the queued time.
     */
    int64_t
    QueuedMs() const;

    /**
     * @brief Set the queued time.
     * @param [in] queued_ms the queued time.
     */
    void
    SetQueuedMs(int64_t queued_ms);

    /**
     * @brief Set the queued time.
     * @param [in] queued_ms the queued time.
     */
    RunningRequestInfo&
    WithQueuedMs(int64_t queued_ms);

    /**
     * @brief How long the request has been running, in milliseconds. In a CancelRequests()
     * answer it is how long the request had been running when it was canceled.
     * @return the elapsed time.
     */
    int64_t
    ElapsedMs() const;

    /**
     * @brief Set the elapsed time.
     * @param [in] elapsed_ms the elapsed time.
     */
    void
    SetElapsedMs(int64_t elapsed_ms);

    /**
     * @brief Set the elapsed time.
     * @param [in] elapsed_ms the elapsed time.
     */
    RunningRequestInfo&
    WithElapsedMs(int64_t elapsed_ms);

    /**
     * @brief The request state: Queued, Running or Canceling. In a CancelRequests() answer
     * it is the state before the cancel, or Canceling when the call repeated an earlier cancel.
     * @return the state.
     */
    const std::string&
    State() const;

    /**
     * @brief Set the request state.
     * @param [in] state the state.
     */
    void
    SetState(const std::string& state);

    /**
     * @brief Set the request state.
     * @param [in] state the state.
     */
    RunningRequestInfo&
    WithState(const std::string& state);

    /**
     * @brief The ids of the proxy tasks the request has created.
     * @return the task ids.
     */
    const std::vector<int64_t>&
    TaskIds() const;

    /**
     * @brief Set the ids of the proxy tasks.
     * @param [in] task_ids the task ids.
     */
    void
    SetTaskIds(const std::vector<int64_t>& task_ids);

    /**
     * @brief Set the ids of the proxy tasks.
     * @param [in] task_ids the task ids.
     */
    void
    SetTaskIds(std::vector<int64_t>&& task_ids);

    /**
     * @brief Set the ids of the proxy tasks.
     * @param [in] task_ids the task ids.
     */
    RunningRequestInfo&
    WithTaskIds(std::vector<int64_t>&& task_ids);

    /**
     * @brief The trace id of the request.
     * @return the trace id.
     */
    const std::string&
    TraceId() const;

    /**
     * @brief Set the trace id.
     * @param [in] trace_id the trace id.
     */
    void
    SetTraceId(const std::string& trace_id);

    /**
     * @brief Set the trace id.
     * @param [in] trace_id the trace id.
     */
    RunningRequestInfo&
    WithTraceId(const std::string& trace_id);

    /**
     * @brief Whether the request can be canceled.
     * @return whether the request can be canceled.
     */
    bool
    Cancellable() const;

    /**
     * @brief Set whether the request can be canceled.
     * @param [in] cancellable whether the request can be canceled.
     */
    void
    SetCancellable(bool cancellable);

    /**
     * @brief Set whether the request can be canceled.
     * @param [in] cancellable whether the request can be canceled.
     */
    RunningRequestInfo&
    WithCancellable(bool cancellable);

 private:
    std::string request_id_;
    int64_t proxy_id_ = 0;
    std::string type_;
    std::string db_name_;
    std::string collection_name_;
    std::string user_;
    std::string client_addr_;
    int64_t nq_ = 0;
    int64_t top_k_ = 0;
    std::string expr_;
    int64_t start_time_ms_ = 0;
    int64_t queued_ms_ = 0;
    int64_t elapsed_ms_ = 0;
    std::string state_;
    std::vector<int64_t> task_ids_;
    std::string trace_id_;
    bool cancellable_ = false;
};

/**
 * @brief RunningRequestInfo objects array
 */
using RunningRequestInfos = std::vector<RunningRequestInfo>;

}  // namespace milvus
