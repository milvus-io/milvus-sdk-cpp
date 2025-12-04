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

#include <memory>
#include <string>

#include "../MilvusConnection.h"
#include "./RpcUtils.h"
#include "common.pb.h"
#include "milvus/Status.h"
#include "milvus/types/ConnectParam.h"
#include "milvus/types/ProgressMonitor.h"
#include "milvus/types/RetryParam.h"

namespace milvus {

class ConnectionHandler {
 public:
    ConnectionHandler() = default;

    Status
    Connect(const ConnectParam& connect_param);

    Status
    Disconnect();

    const MilvusConnectionPtr&
    GetConnection() const;

    Status
    SetRpcDeadlineMs(uint64_t timeout_ms);

    uint64_t
    GetRpcDeadlineMs() const;

    Status
    SetRetryParam(const RetryParam& retry_param);

    const RetryParam&
    GetRetryParam() const;

    Status
    UseDatabase(const std::string& db_name);

    std::string
    CurrentDbName(const std::string& overwrite_db_name) const;

    // This interface is not exposed to users
    Status
    GetLoadingProgress(const std::string& db_name, const std::string& collection_name,
                       const std::set<std::string> partition_names, uint32_t& progress);

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

 private:
    /**
     * @brief template for public api call
     *        validate -> pre -> rpc -> wait_for_status -> post
     */
    template <typename Request, typename Response>
    Status
    apiHandler(const std::function<Status(void)>& validate, std::function<Status(Request&)> pre,
               Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&),
               std::function<Status(const Response&)> wait_for_status, std::function<Status(const Response&)> post) {
        if (connection_ == nullptr) {
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
        uint64_t timeout = connection_->GetConnectParam().RpcDeadlineMs();
        auto func = std::bind(rpc, connection_.get(), rpc_request, std::placeholders::_1, GrpcOpts{timeout});
        auto caller = [&func, &rpc_response]() { return func(rpc_response); };
        auto status = Retry(caller, retry_param_);
        if (!status.IsOk()) {
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
    MilvusConnectionPtr connection_;
    RetryParam retry_param_;
};

}  // namespace milvus
