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

#include "ConnectionHandler.h"

#include <chrono>
#include <thread>

#include "GlobalClusterUtils.h"
#include "TopologyRefresher.h"

namespace milvus {

ConnectionHandler::ConnectionHandler() = default;

ConnectionHandler::~ConnectionHandler() {
    stopGlobalRefresher();
}

Status
ConnectionHandler::Connect(const ConnectParam& connect_param) {
    // Keep the candidate private until its handshake succeeds, but do not hold mtx_ while
    // AttachChannel waits for an in-flight command handler. The lifecycle lock also fences
    // concurrent explicit connects, disconnects, database switches, and global failovers.
    // It must remain fail-fast: a command handler holds telemetry's command_mutex and may
    // re-enter a lifecycle API while an external lifecycle call is waiting in AttachChannel.
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mtx_, std::try_to_lock);
    if (!lifecycle_lock.owns_lock()) {
        return {StatusCode::CLIENT_BUSY, "Connection lifecycle change is already in progress"};
    }

    // Snapshot only reusable resources. The published connection and global state remain untouched
    // until the candidate handshake succeeds, so every failure is a no-op from callers' perspective.
    ClientTelemetryManagerPtr reusable_telemetry;
    MilvusConnectionPtr old_connection;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        reusable_telemetry = telemetry_;
        old_connection = connection_;
        if (old_connection != nullptr) {
            auto current_telemetry = old_connection->GetTelemetry();
            if (current_telemetry != nullptr) {
                reusable_telemetry = std::move(current_telemetry);
            }
        }
    }

    bool is_global = GlobalClusterUtils::IsGlobalEndpoint(connect_param.Uri());
    ConnectParam primary_param = connect_param;
    GlobalTopology initial_topology;
    if (is_global) {
        // fetch the topology outside the lock: it can block for tens of seconds (3 attempts with
        // 10s timeouts plus backoff) and must not stall concurrent RPC operations that snapshot
        // the connection under mtx_
        auto status = GlobalClusterUtils::FetchTopology(connect_param.Uri(), connect_param.Token(), initial_topology);
        if (!status.IsOk()) {
            return status;
        }
        const ClusterInfo* primary = initial_topology.Primary();
        if (primary == nullptr) {
            return {StatusCode::SERVER_FAILED, "No primary (writable) cluster found in global topology"};
        }
        primary_param.SetUri(
            GlobalClusterUtils::BuildPrimaryUri(connect_param.Uri(), connect_param.TlsEnabled(), primary->Endpoint()));
    }

    std::unique_ptr<TopologyRefresher> new_refresher;
    if (is_global) {
        new_refresher = std::make_unique<TopologyRefresher>(
            connect_param.Uri(), connect_param.Token(), initial_topology.Version(), std::chrono::seconds(300),
            [this](const GlobalTopology& topology, const std::function<bool()>& should_stop) {
                return reconnectToPrimary(topology, should_stop);
            });
        // Starting a private refresher cannot observe or mutate the live lifecycle before commit:
        // its first refresh waits for the configured interval. If thread creation throws, the old
        // published state is still intact and the exception is converted to a Status below. Starting
        // it before the candidate attaches shared telemetry also keeps that handoff as the final
        // infallible step before publication.
        try {
            new_refresher->Start();
        } catch (const std::exception& exception) {
            return {StatusCode::UNKNOWN_ERROR,
                    std::string("Failed to start global topology refresher: ") + exception.what()};
        }
    }

    auto new_connection = std::make_shared<MilvusConnection>();
    auto status = new_connection->Connect(primary_param, telemetry_client_id_, reusable_telemetry,
                                          is_global ? connect_param.Uri() : "");
    if (!status.IsOk()) {
        if (new_refresher != nullptr) {
            new_refresher->Stop();
        }
        return status;
    }

    auto telemetry = new_connection->GetTelemetry();
    std::unique_ptr<TopologyRefresher> old_refresher;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        old_connection = connection_;
        old_refresher = std::move(global_refresher_);
        global_mode_ = is_global;
        global_endpoint_ = is_global ? connect_param.Uri() : std::string{};
        if (telemetry != nullptr) {
            telemetry_ = telemetry;
            telemetry_client_id_ = telemetry->ClientId();
        }
        connection_ = new_connection;

        // Commit global-cluster state only after the primary connect succeeds.
        if (is_global) {
            global_connect_param_ = connect_param;
            global_refresher_ = std::move(new_refresher);
        }
    }

    // The old refresher can be inside its callback. It cannot wait for lifecycle_mtx_ because
    // reconnectToPrimary also uses try_to_lock, so Stop()/join is safe while this lifecycle owns it.
    if (old_refresher != nullptr) {
        old_refresher->Stop();
    }
    if (old_connection != nullptr) {
        const bool shares_telemetry = telemetry != nullptr && old_connection->GetTelemetry() == telemetry;
        old_connection->Disconnect(!shares_telemetry);
    }
    return Status::OK();
}

Status
ConnectionHandler::Disconnect() {
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mtx_, std::try_to_lock);
    if (!lifecycle_lock.owns_lock()) {
        return {StatusCode::CLIENT_BUSY, "Connection lifecycle change is already in progress"};
    }

    // stop the refresher without holding the lock; callbacks see global_mode_==false and no-op
    stopGlobalRefresher();

    MilvusConnectionPtr connection;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        connection = std::move(connection_);
        if (connection == nullptr) {
            return Status::OK();
        }
        auto telemetry = connection->GetTelemetry();
        if (telemetry != nullptr) {
            telemetry_ = std::move(telemetry);
        }
    }
    return connection->Disconnect();
}

void
ConnectionHandler::stopGlobalRefresher() {
    std::unique_ptr<TopologyRefresher> refresher;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        // mark global mode off first so an in-flight callback cannot reconnect during teardown
        global_mode_ = false;
        refresher = std::move(global_refresher_);
    }
    // destroy (joins) the refresher outside the lock
    if (refresher != nullptr) {
        refresher->Stop();
    }
}

void
ConnectionHandler::TriggerGlobalRefresh() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (global_refresher_ != nullptr) {
        global_refresher_->TriggerRefresh();
    }
}

bool
ConnectionHandler::reconnectToPrimary(const GlobalTopology& topology, const std::function<bool()>& should_stop) {
    // A refresher callback must never wait behind Connect()/Disconnect() while those methods
    // are stopping and joining the refresher thread.
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mtx_, std::try_to_lock);
    if (!lifecycle_lock.owns_lock()) {
        return false;
    }

    const ClusterInfo* primary = topology.Primary();
    if (primary == nullptr) {
        // no writable cluster in this topology; report failure so the refresher retries the
        // same version next interval instead of committing it and going silent
        return false;
    }

    ConnectParam primary_param;
    ClientTelemetryManagerPtr reusable_telemetry;
    std::string telemetry_client_id;
    std::string telemetry_logical_endpoint;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!global_mode_) {
            return true;
        }

        std::string new_primary_uri = GlobalClusterUtils::BuildPrimaryUri(
            global_connect_param_.Uri(), global_connect_param_.TlsEnabled(), primary->Endpoint());
        // a topology version bump with the same primary endpoint needs no reconnect
        // (matching Java/pymilvus behavior)
        if (connection_ != nullptr && connection_->GetConnectParam().Uri() == new_primary_uri) {
            return true;
        }

        primary_param = global_connect_param_;
        primary_param.SetUri(new_primary_uri);
        // re-apply configuration applied after Connect() so failover does not silently lose it
        if (connection_ != nullptr) {
            primary_param.SetDbName(connection_->GetConnectParam().DbName());
            primary_param.SetRpcDeadlineMs(connection_->GetConnectParam().RpcDeadlineMs());
            reusable_telemetry = connection_->GetTelemetry();
        }
        if (reusable_telemetry == nullptr) {
            reusable_telemetry = telemetry_;
        }
        telemetry_client_id = telemetry_client_id_;
        telemetry_logical_endpoint = global_endpoint_;
    }

    // abort promptly when the refresher is stopping rather than starting a fresh gRPC connect
    if (should_stop && should_stop()) {
        return false;
    }

    // build + connect the new primary outside the lock: MilvusConnection::Connect() blocks in
    // WaitForConnected() and the Connect RPC for up to ~2x ConnectTimeout, and holding mtx_ that
    // long would stall every other SDK operation that snapshots the connection.
    auto new_connection = std::make_shared<MilvusConnection>();
    auto status =
        new_connection->Connect(primary_param, telemetry_client_id, reusable_telemetry, telemetry_logical_endpoint);
    if (!status.IsOk()) {
        // keep the existing connection; report failure so the refresher retries the same version
        return false;
    }

    auto new_telemetry = new_connection->GetTelemetry();
    MilvusConnectionPtr old_connection;
    bool discard_candidate = false;
    bool stop_candidate_telemetry = true;
    bool reconnect_result = true;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!global_mode_) {
            // disconnected while reconnecting; discard the unused candidate connection
            discard_candidate = true;
            stop_candidate_telemetry = new_telemetry == nullptr || telemetry_ != new_telemetry;
        } else if (connection_ != nullptr) {
            // Re-read live configuration before swapping the candidate.
            const ConnectParam& live = connection_->GetConnectParam();
            new_connection->GetConnectParam().SetRpcDeadlineMs(live.RpcDeadlineMs());
            if (new_connection->GetConnectParam().DbName() != live.DbName()) {
                // the database changed while reconnecting; drop the stale candidate and retry
                discard_candidate = true;
                stop_candidate_telemetry = new_telemetry == nullptr || connection_->GetTelemetry() != new_telemetry;
                reconnect_result = false;
            }
        }

        if (!discard_candidate) {
            old_connection = connection_;
            connection_ = new_connection;
            if (new_telemetry != nullptr) {
                telemetry_ = new_telemetry;
                telemetry_client_id_ = new_telemetry->ClientId();
            }
        }
    }

    if (discard_candidate) {
        new_connection->Disconnect(stop_candidate_telemetry);
        return reconnect_result;
    }
    if (old_connection != nullptr) {
        const bool shares_telemetry = new_telemetry != nullptr && old_connection->GetTelemetry() == new_telemetry;
        old_connection->Disconnect(!shares_telemetry);
    }
    return true;
}

MilvusConnectionPtr
ConnectionHandler::GetConnection() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return connection_;
}

ClientTelemetryManagerPtr
ConnectionHandler::GetTelemetry() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return connection_ == nullptr ? telemetry_ : connection_->GetTelemetry();
}

Status
ConnectionHandler::SetRpcDeadlineMs(uint64_t timeout_ms) {
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mtx_, std::try_to_lock);
    if (!lifecycle_lock.owns_lock()) {
        return {StatusCode::CLIENT_BUSY, "Connection lifecycle change is already in progress"};
    }
    std::lock_guard<std::mutex> lock(mtx_);
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not created!"};
    }
    connection_->GetConnectParam().SetRpcDeadlineMs(timeout_ms);
    return Status::OK();
}

uint64_t
ConnectionHandler::GetRpcDeadlineMs() const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (connection_ != nullptr) {
        return connection_->GetConnectParam().RpcDeadlineMs();
    }
    return 0;
}

Status
ConnectionHandler::SetRetryParam(const RetryParam& retry_param) {
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mtx_, std::try_to_lock);
    if (!lifecycle_lock.owns_lock()) {
        return {StatusCode::CLIENT_BUSY, "Connection lifecycle change is already in progress"};
    }
    std::lock_guard<std::mutex> lock(mtx_);
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not created!"};
    }
    retry_param_ = retry_param;
    return Status::OK();
}

RetryParam
ConnectionHandler::GetRetryParam() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return retry_param_;
}

Status
ConnectionHandler::UseDatabase(const std::string& db_name) {
    std::unique_lock<std::mutex> lifecycle_lock(lifecycle_mtx_, std::try_to_lock);
    if (!lifecycle_lock.owns_lock()) {
        return {StatusCode::CLIENT_BUSY, "Connection lifecycle change is already in progress"};
    }
    auto connection = GetConnection();
    return connection == nullptr ? Status::OK() : connection->UseDatabase(db_name);
}

std::string
ConnectionHandler::CurrentDbName(const std::string& overwrite_db_name) const {
    // if a db name is specified for rpc interface, use this name
    if (!overwrite_db_name.empty()) {
        return overwrite_db_name;
    }
    // no db name is specified, use the current db name used by this connection
    std::lock_guard<std::mutex> lock(mtx_);
    if (connection_ != nullptr) {
        const ConnectParam& param = connection_->GetConnectParam();
        // Preserve an empty database for the RPC. Cache keys normalize it to "default" separately.
        return param.DbName();
    }
    return "";
}

std::string
ConnectionHandler::CurrentEndpoint() const {
    std::lock_guard<std::mutex> lock(mtx_);
    // Keep cache keys scoped to the logical global-cluster endpoint so schemas/timestamps recorded
    // before a primary failover remain reachable (matching the Java SDK behavior).
    if (global_mode_ && !global_endpoint_.empty()) {
        return global_endpoint_;
    }
    if (connection_ == nullptr) {
        return "";
    }
    return connection_->GetConnectParam().Uri();
}

Status
ConnectionHandler::GetLoadingProgress(const std::string& db_name, const std::string& collection_name,
                                      const std::set<std::string>& partition_names, uint32_t& progress,
                                      uint32_t& refresh_progress, uint64_t rpc_timeout_ms) {
    MilvusConnectionPtr connection;
    uint64_t timeout = 0;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (connection_ == nullptr) {
            return {StatusCode::NOT_CONNECTED, "Connection is not created!"};
        }
        connection = connection_;
        timeout = connection_->GetConnectParam().RpcDeadlineMs();
    }

    proto::milvus::GetLoadingProgressRequest progress_req;
    progress_req.set_db_name(db_name);
    progress_req.set_collection_name(collection_name);
    for (const auto& partition_name : partition_names) {
        progress_req.add_partition_names(partition_name);
    }
    proto::milvus::GetLoadingProgressResponse progress_resp;
    if (rpc_timeout_ms > 0 && (timeout == 0 || rpc_timeout_ms < timeout)) {
        timeout = rpc_timeout_ms;
    }

    auto status = connection->GetLoadingProgress(progress_req, progress_resp, GrpcOpts{timeout});
    if (!status.IsOk()) {
        return status;
    }
    progress = static_cast<uint32_t>(progress_resp.progress());
    refresh_progress = static_cast<uint32_t>(progress_resp.refresh_progress());
    return Status::OK();
}

Status
ConnectionHandler::WaitForStatus(const std::function<Status(Progress&)>& query_function,
                                 const ProgressMonitor& progress_monitor) {
    // no need to check
    if (progress_monitor.CheckTimeout() == 0) {
        return Status::OK();
    }

    std::chrono::time_point<std::chrono::steady_clock> started = std::chrono::steady_clock::now();

    auto calculated_next_wait = started;
    auto wait_milliseconds = progress_monitor.CheckTimeout() * 1000;
    auto wait_interval = progress_monitor.CheckInterval();
    auto final_timeout = started + std::chrono::milliseconds{wait_milliseconds};
    while (true) {
        calculated_next_wait += std::chrono::milliseconds{wait_interval};
        auto next_wait = std::min(calculated_next_wait, final_timeout);
        std::this_thread::sleep_until(next_wait);

        Progress current_progress;
        auto status = query_function(current_progress);
        // if the internal check function failed, return error
        if (!status.IsOk()) {
            return status;
        }

        // notify progress
        progress_monitor.DoProgress(current_progress);

        // if progress all done, break the circle
        if (current_progress.Done()) {
            return status;
        }

        // if time to deadline, return timeout error
        if (next_wait >= final_timeout) {
            return Status{StatusCode::TIMEOUT, "time out"};
        }
    }
}

}  // namespace milvus
