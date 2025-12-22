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

#include <thread>

namespace milvus {

Status
ConnectionHandler::Connect(const ConnectParam& connect_param) {
    if (connection_ != nullptr) {
        connection_->Disconnect();
    }

    // TODO: check connect parameter
    connection_ = std::make_shared<MilvusConnection>();
    return connection_->Connect(connect_param);
}

Status
ConnectionHandler::Disconnect() {
    if (connection_ != nullptr) {
        return connection_->Disconnect();
    }
    return Status::OK();
}

const MilvusConnectionPtr&
ConnectionHandler::GetConnection() const {
    return connection_;
}

Status
ConnectionHandler::SetRpcDeadlineMs(uint64_t timeout_ms) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not created!"};
    }
    connection_->GetConnectParam().SetRpcDeadlineMs(timeout_ms);
    return Status::OK();
}

uint64_t
ConnectionHandler::GetRpcDeadlineMs() const {
    if (connection_ != nullptr) {
        return connection_->GetConnectParam().RpcDeadlineMs();
    }
    return 0;
}

Status
ConnectionHandler::SetRetryParam(const RetryParam& retry_param) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not created!"};
    }
    retry_param_ = retry_param;
    return Status::OK();
}

const RetryParam&
ConnectionHandler::GetRetryParam() const {
    return retry_param_;
}

Status
ConnectionHandler::UseDatabase(const std::string& db_name) {
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
    if (connection_ != nullptr) {
        const ConnectParam& param = connection_->GetConnectParam();
        return param.DbName();
    }
    return "";
}

Status
ConnectionHandler::GetLoadingProgress(const std::string& db_name, const std::string& collection_name,
                                      const std::set<std::string> partition_names, uint32_t& progress) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not created!"};
    }

    proto::milvus::GetLoadingProgressRequest progress_req;
    progress_req.set_db_name(db_name);
    progress_req.set_collection_name(collection_name);
    for (const auto& partition_name : partition_names) {
        progress_req.add_partition_names(partition_name);
    }
    proto::milvus::GetLoadingProgressResponse progress_resp;
    uint64_t timeout = GetRpcDeadlineMs();

    auto status = connection_->GetLoadingProgress(progress_req, progress_resp, GrpcOpts{timeout});
    if (!status.IsOk()) {
        return status;
    }
    progress = static_cast<uint32_t>(progress_resp.progress());
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
