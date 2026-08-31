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
#include <memory>
#include <mutex>
#include <string>

#include "../MilvusConnection.h"
#include "../types/GlobalCluster.h"
#include "./RpcUtils.h"
#include "common.pb.h"
#include "milvus/Status.h"
#include "milvus/types/ConnectParam.h"
#include "milvus/types/ProgressMonitor.h"
#include "milvus/types/RetryParam.h"

namespace milvus {

class TopologyRefresher;

class ConnectionHandler {
 public:
    ConnectionHandler();

    ~ConnectionHandler();

    Status
    Connect(const ConnectParam& connect_param);

    Status
    Disconnect();

    MilvusConnectionPtr
    GetConnection() const;

    ClientTelemetryManagerPtr
    GetTelemetry() const;

    Status
    SetRpcDeadlineMs(uint64_t timeout_ms);

    uint64_t
    GetRpcDeadlineMs() const;

    Status
    SetRetryParam(const RetryParam& retry_param);

    RetryParam
    GetRetryParam() const;

    Status
    UseDatabase(const std::string& db_name);

    /**
     * @brief Reconnect using the given username/password credentials. Must be serialized with
     * other operations, matching UseDatabase.
     */
    Status
    ResetUserCredentials(const std::string& username, const std::string& password);

    std::string
    CurrentDbName(const std::string& overwrite_db_name) const;

    std::string
    CurrentEndpoint() const;

    // This interface is not exposed to users
    Status
    GetLoadingProgress(const std::string& db_name, const std::string& collection_name,
                       const std::set<std::string>& partition_names, uint32_t& progress, uint32_t& refresh_progress,
                       uint64_t rpc_timeout_ms = 0);

    /**
     * @brief Trigger an immediate global-cluster topology refresh (debounced). No-op for
     * non-global-cluster connections. Called when an RPC hits UNAVAILABLE.
     */
    void
    TriggerGlobalRefresh();

    /**
     * Internal wait for status query done.
     *
     * @param [in] query_function one time query for return Status, return TIMEOUT status if not done
     * @param [in] progress_monitor timeout setting for waiting progress
     * @return Status, the final status
     */
    static Status
    WaitForStatus(const std::function<Status(Progress&)>& query_function, const ProgressMonitor& progress_monitor);

    /**
     * @brief template for public api call
     */
    template <typename Request, typename Response>
    Status
    Invoke(std::function<Status(void)> validate, std::function<Status(Request&)> pre,
           Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&),
           std::function<Status(const Response&)> post) {
        return apiHandler(validate, pre, rpc, std::function<Status(const Response&)>{}, post);
    }

    /**
     * @brief template for public api call
     */
    template <typename Request, typename Response>
    Status
    Invoke(std::function<Status(void)> validate, std::function<Status(Request&)> pre,
           Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&)) {
        return apiHandler(validate, pre, rpc, std::function<Status(const Response&)>{},
                          std::function<Status(const Response&)>{});
    }

    /**
     * @brief template for public api call
     */
    template <typename Request, typename Response>
    Status
    Invoke(std::function<Status(Request&)> pre,
           Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&),
           std::function<Status(const Response&)> post) {
        return apiHandler(std::function<Status(void)>{}, pre, rpc, std::function<Status(const Response&)>{}, post);
    }

    /**
     * @brief template for public api call
     */
    template <typename Request, typename Response>
    Status
    Invoke(std::function<Status(Request&)> pre,
           Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&)) {
        return apiHandler(std::function<Status(void)>{}, pre, rpc, std::function<Status(const Response&)>{},
                          std::function<Status(const Response&)>{});
    }

    template <typename Request, typename Response>
    Status
    Invoke(const std::function<Status(void)>& validate, std::function<Status(Request&)> pre,
           Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&),
           std::function<Status(const Response&)> wait_for_status, std::function<Status(const Response&)> post) {
        return apiHandler(validate, pre, rpc, wait_for_status, post);
    }

    template <typename Request, typename Response>
    Status
    InvokeWithRpcTimeout(uint64_t rpc_timeout_ms, std::function<Status(Request&)> pre,
                         Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&),
                         std::function<Status(const Response&)> post) {
        return apiHandler(std::function<Status(void)>{}, pre, rpc, std::function<Status(const Response&)>{}, post,
                          rpc_timeout_ms);
    }

    template <typename Request, typename Response>
    Status
    InvokeWithRpcTimeout(uint64_t rpc_timeout_ms, std::function<Status(Request&)> pre,
                         Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&)) {
        return apiHandler(std::function<Status(void)>{}, pre, rpc, std::function<Status(const Response&)>{},
                          std::function<Status(const Response&)>{}, rpc_timeout_ms);
    }

    template <typename Request, typename Response>
    Status
    InvokeWithRpcTimeout(uint64_t rpc_timeout_ms, const std::function<Status(void)>& validate,
                         std::function<Status(Request&)> pre,
                         Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&),
                         std::function<Status(const Response&)> wait_for_status,
                         std::function<Status(const Response&)> post) {
        return apiHandler(validate, pre, rpc, wait_for_status, post, rpc_timeout_ms);
    }

 private:
    /**
     * @brief Reconnect to a new primary endpoint after a topology change.
     * Builds the new connection outside the mutex and only swaps it under the lock.
     * @param should_stop aborts the reconnect promptly (e.g. when the refresher is stopping)
     * @return true when the topology change was handled (the version can be committed), false
     * when the reconnect failed or no writable primary exists (so the refresher retries the
     * same version next interval).
     */
    bool
    reconnectToPrimary(const GlobalTopology& topology, const std::function<bool()>& should_stop);

    /**
     * @brief Stop the global-cluster refresher without holding the connection mutex, so an
     * in-flight callback waiting on the mutex cannot deadlock the join. Marks global mode off
     * first so any in-flight callback becomes a no-op.
     */
    void
    stopGlobalRefresher();

    /**
     * @brief template for public api call
     *        validate -> pre -> rpc -> wait_for_status -> post
     */
    template <typename Request, typename Response>
    Status
    apiHandler(const std::function<Status(void)>& validate, std::function<Status(Request&)> pre,
               Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&),
               std::function<Status(const Response&)> wait_for_status, std::function<Status(const Response&)> post,
               uint64_t rpc_timeout_ms = 0) {
        MilvusConnectionPtr connection;
        RetryParam retry_param;
        uint64_t timeout = 0;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (connection_ == nullptr) {
                return {StatusCode::NOT_CONNECTED, "Connection is not created!"};
            }
            connection = connection_;
            retry_param = retry_param_;
            timeout = connection_->GetConnectParam().RpcDeadlineMs();
        }
        if (connection == nullptr) {
            return {StatusCode::NOT_CONNECTED, "Connection is not created!"};
        }

        // validate input
        if (validate) {
            auto status = validate();
            if (!status.IsOk()) {
                return status;
            }
        }

        // construct rpc request
        Request rpc_request;
        if (pre) {
            auto status = pre(rpc_request);
            if (!status.IsOk()) {
                return status;
            }
        }

        // call rpc interface
        Response rpc_response;
        // the timeout value can be changed by MilvusClient::SetRpcDeadlineMs()
        if (rpc_timeout_ms > 0 && (timeout == 0 || rpc_timeout_ms < timeout)) {
            timeout = rpc_timeout_ms;
        }
        auto func = std::bind(rpc, connection.get(), rpc_request, std::placeholders::_1, GrpcOpts{timeout});
        auto caller = [&func, &rpc_response]() { return func(rpc_response); };
        auto status = Retry(caller, retry_param);
        if (!status.IsOk()) {
            // A global-cluster primary may have become unreachable (gRPC UNAVAILABLE) or may be
            // actively rejecting writes during a failover window (STREAMING_CODE_REPLICATE_VIOLATION,
            // surfaced in the server's reason string). Trigger a topology refresh for either, matching
            // pymilvus's handle_error() which keys off grpc.UNAVAILABLE and the replicate-violation
            // message text.
            if (status.RpcErrCode() == ::grpc::StatusCode::UNAVAILABLE ||
                status.Message().find("STREAMING_CODE_REPLICATE_VIOLATION") != std::string::npos) {
                TriggerGlobalRefresh();
            }
            // response's status already checked in connection class
            return status;
        }

        // wait loop
        if (wait_for_status) {
            status = wait_for_status(rpc_response);
            if (!status.IsOk()) {
                return status;
            }
        }

        // process results
        if (post) {
            status = post(rpc_response);
            if (!status.IsOk()) {
                return status;
            }
        }
        return Status::OK();
    }

 private:
    // Lifecycle lock order and re-entrancy contract:
    // 1. Every state-mutating lifecycle entry acquires lifecycle_mtx_ with try_to_lock.
    // 2. While it owns lifecycle_mtx_, it may briefly acquire mtx_ to snapshot or commit state.
    // 3. mtx_ is never held across network I/O, TopologyRefresher::Stop(), telemetry Stop(),
    //    AttachChannel(), or ProcessCommands().
    // A telemetry command handler already owns command_mutex and may re-enter a lifecycle API.
    // Fail-fast lifecycle acquisition prevents a command_mutex -> lifecycle_mtx_ wait from
    // forming a cycle with an external lifecycle_mtx_ -> AttachChannel(command_mutex) handoff.
    mutable std::mutex lifecycle_mtx_;
    mutable std::mutex mtx_;
    MilvusConnectionPtr connection_;
    ClientTelemetryManagerPtr telemetry_;
    RetryParam retry_param_;
    std::string telemetry_client_id_;

    // global-cluster state
    bool global_mode_{false};
    std::string global_endpoint_;
    ConnectParam global_connect_param_;
    std::unique_ptr<TopologyRefresher> global_refresher_;
};

}  // namespace milvus
