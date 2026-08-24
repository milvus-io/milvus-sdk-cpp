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
    // Snapshot the previous global-cluster state so a failed handshake can restore it: the old
    // primary connection_ is kept until the new handshake succeeds, and tearing down the refresher
    // before the attempt would otherwise drop global tracking from a still-usable connection.
    bool was_global = false;
    std::string prior_endpoint;
    ConnectParam prior_connect_param;
    std::unique_ptr<TopologyRefresher> prior_refresher;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        was_global = global_mode_;
        prior_endpoint = global_endpoint_;
        prior_connect_param = global_connect_param_;
        prior_refresher = std::move(global_refresher_);
        // mark global mode off first so an in-flight callback cannot reconnect during teardown
        global_mode_ = false;
    }
    // join the refresher thread outside the lock (it may be inside reconnectToPrimary waiting on mtx_)
    if (prior_refresher != nullptr) {
        prior_refresher->Stop();
    }

    // Restore the prior global state on the failure path (only when a previous connection is still
    // live), so cache-key scope and automatic failover are preserved for the old primary.
    auto restore = [&]() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (connection_ == nullptr) {
            return;
        }
        global_mode_ = was_global;
        global_endpoint_ = prior_endpoint;
        global_connect_param_ = prior_connect_param;
        global_refresher_ = std::move(prior_refresher);
        if (global_refresher_ != nullptr) {
            global_refresher_->Start();
        }
    };

    bool is_global = GlobalClusterUtils::IsGlobalEndpoint(connect_param.Uri());
    ConnectParam primary_param = connect_param;
    GlobalTopology initial_topology;
    if (is_global) {
        // fetch the topology outside the lock: it can block for tens of seconds (3 attempts with
        // 10s timeouts plus backoff) and must not stall concurrent RPC operations that snapshot
        // the connection under mtx_
        auto status = GlobalClusterUtils::FetchTopology(connect_param.Uri(), connect_param.Token(), initial_topology);
        if (!status.IsOk()) {
            restore();
            return status;
        }
        const ClusterInfo* primary = initial_topology.Primary();
        if (primary == nullptr) {
            restore();
            return {StatusCode::SERVER_FAILED, "No primary (writable) cluster found in global topology"};
        }
        primary_param.SetUri(
            GlobalClusterUtils::BuildPrimaryUri(connect_param.Uri(), connect_param.TlsEnabled(), primary->Endpoint()));
    }

    // Serialize the connection handshake with lifecycle and configuration mutations. The candidate
    // connection remains private until it succeeds, but setters must not update the current
    // connection and then be overwritten by the successful swap below.
    MilvusConnectionPtr new_connection;
    Status status;
    {
        std::lock_guard<std::mutex> lock(mtx_);

        global_mode_ = false;
        global_endpoint_.clear();

        new_connection = std::make_shared<MilvusConnection>();
        status = new_connection->Connect(primary_param);
        if (!status.IsOk()) {
            // fall through to restore the prior global state after releasing the lock
        } else {
            if (connection_ != nullptr) {
                connection_->Disconnect();
            }
            connection_ = std::move(new_connection);

            // commit the global-cluster state only after the primary connect succeeded, so a failed
            // Connect() leaves the handler in non-global mode (consistent with connection_ == null)
            if (is_global) {
                global_mode_ = true;
                global_endpoint_ = connect_param.Uri();
                global_connect_param_ = connect_param;
                global_refresher_ = std::make_unique<TopologyRefresher>(
                    global_endpoint_, global_connect_param_.Token(), initial_topology.Version(),
                    std::chrono::seconds(300),
                    [this](const GlobalTopology& topology, const std::function<bool()>& should_stop) {
                        return reconnectToPrimary(topology, should_stop);
                    });
                global_refresher_->Start();
            }
        }
    }
    if (!status.IsOk()) {
        restore();
        return status;
    }
    return Status::OK();
}

Status
ConnectionHandler::Disconnect() {
    // stop the refresher without holding the lock; callbacks see global_mode_==false and no-op
    stopGlobalRefresher();

    std::lock_guard<std::mutex> lock(mtx_);
    if (connection_ != nullptr) {
        return connection_->Disconnect();
    }
    return Status::OK();
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
    const ClusterInfo* primary = topology.Primary();
    if (primary == nullptr) {
        // no writable cluster in this topology; report failure so the refresher retries the
        // same version next interval instead of committing it and going silent
        return false;
    }

    ConnectParam primary_param;
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
        }
    }

    // abort promptly when the refresher is stopping rather than starting a fresh gRPC connect
    if (should_stop && should_stop()) {
        return false;
    }

    // build + connect the new primary outside the lock: MilvusConnection::Connect() blocks in
    // WaitForConnected() and the Connect RPC for up to ~2x ConnectTimeout, and holding mtx_ that
    // long would stall every other SDK operation that snapshots the connection.
    auto new_connection = std::make_shared<MilvusConnection>();
    auto status = new_connection->Connect(primary_param);
    if (!status.IsOk()) {
        // keep the existing connection; report failure so the refresher retries the same version
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!global_mode_) {
            // disconnected while reconnecting; discard the unused candidate connection
            new_connection->Disconnect();
            return true;
        }
        // re-read live configuration in case SetRpcDeadlineMs()/UseDatabase() ran while the
        // candidate was being built outside the lock, so it is not silently dropped on swap
        if (connection_ != nullptr) {
            const ConnectParam& live = connection_->GetConnectParam();
            new_connection->GetConnectParam().SetRpcDeadlineMs(live.RpcDeadlineMs());
            if (new_connection->GetConnectParam().DbName() != live.DbName()) {
                // the database changed while reconnecting; drop the stale candidate and retry
                new_connection->Disconnect();
                return false;
            }
        }
        auto old_connection = connection_;
        connection_ = std::move(new_connection);
        if (old_connection != nullptr) {
            old_connection->Disconnect();
        }
    }
    return true;
}

MilvusConnectionPtr
ConnectionHandler::GetConnection() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return connection_;
}

Status
ConnectionHandler::SetRpcDeadlineMs(uint64_t timeout_ms) {
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
    std::lock_guard<std::mutex> lock(mtx_);
    if (connection_ != nullptr) {
        return connection_->UseDatabase(db_name);
    }

    return Status::OK();
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
