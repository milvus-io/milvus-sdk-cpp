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

#include "RpcUtils.h"

#include <grpcpp/channel.h>

#include <thread>

#include "GtsDict.h"
#include "common.pb.h"

namespace milvus {
Status
Retry(std::function<Status(void)> caller, const RetryParam& retry_param) {
    auto max_retry_times = retry_param.MaxRetryTimes();
    // no retry, call the method
    if (max_retry_times <= 1) {
        return caller();
    }

    auto begin = GetNowMs();
    auto max_timeout_ms = retry_param.MaxRetryTimeoutMs();
    auto is_timeout = [&begin, &max_timeout_ms]() {
        auto current = GetNowMs();
        auto cost = (current - begin);
        if (max_timeout_ms > 0 && cost >= max_timeout_ms) {
            return true;
        }
        return false;
    };

    auto retry_interval_ms = retry_param.InitialBackOffMs();
    for (auto k = 1; k <= max_retry_times; k++) {
        auto status = caller();
        if (status.IsOk()) {
            return status;
        }

        // the following rpc error codes cannot be retried
        auto rpc_code = status.RpcErrCode();
        if (rpc_code == ::grpc::StatusCode::DEADLINE_EXCEEDED || rpc_code == ::grpc::StatusCode::PERMISSION_DENIED ||
            rpc_code == ::grpc::StatusCode::UNAUTHENTICATED || rpc_code == ::grpc::StatusCode::INVALID_ARGUMENT ||
            rpc_code == ::grpc::StatusCode::ALREADY_EXISTS || rpc_code == ::grpc::StatusCode::RESOURCE_EXHAUSTED ||
            rpc_code == ::grpc::StatusCode::UNIMPLEMENTED) {
            std::string msg = "Encounter rpc error that cannot be retried, reason: " + status.Message();
            auto code =
                (rpc_code == ::grpc::StatusCode::DEADLINE_EXCEEDED) ? StatusCode::TIMEOUT : StatusCode::RPC_FAILED;
            return Status{code, msg, rpc_code, status.ServerCode(), status.LegacyServerCode()};
        }

        // for server-side returned error, only retry for rate limit
        // error codes of v2.2, LegacyServerCode value is 49
        // error codes of v2.3, rate limit error value is 8
        if (retry_param.RetryOnRateLimit() &&
            (status.LegacyServerCode() == static_cast<int32_t>(proto::common::ErrorCode::RateLimit) ||
             status.ServerCode() == 8)) {
            // can be retried
        } else if (!status.IsOk()) {
            // server-side error cannot be retried, exit retry, return the error
            return status;
        }

        if (k >= max_retry_times) {
            // finish retry loop
            std::string msg = std::to_string(max_retry_times) + " retry times, stop retry";
            return Status{StatusCode::TIMEOUT, msg, rpc_code, status.ServerCode(), status.LegacyServerCode()};
        } else {
            // sleep for interval
            // TODO: print log
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_interval_ms));
            // reset the next interval value
            retry_interval_ms = retry_interval_ms * retry_param.BackOffMultiplier();
            if (retry_interval_ms > retry_param.MaxBackOffMs()) {
                retry_interval_ms = retry_param.MaxBackOffMs();
            }
        }

        if (is_timeout()) {
            std::string msg = "Retry timeout: " + std::to_string(max_timeout_ms) +
                              " max_retry: " + std::to_string(max_retry_times) + " retries: " + std::to_string(k + 1) +
                              " reason: " + status.Message();
            return Status{StatusCode::TIMEOUT, msg, rpc_code, status.ServerCode(), status.LegacyServerCode()};
        }
    }

    // theorectically this line will not be hit
    return Status::OK();
}

}  // namespace milvus
