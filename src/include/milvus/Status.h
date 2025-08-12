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

#include <string>

/**
 * @brief Milvus SDK namespace
 */
namespace milvus {

/**
 * @brief Status code for SDK interface return
 */
enum class StatusCode {
    OK = 0,

    // system error section
    UNKNOWN_ERROR = 1,
    NOT_SUPPORTED,
    NOT_CONNECTED,

    // function error section
    INVALID_AGUMENT = 1000,
    RPC_FAILED,
    SERVER_FAILED,
    TIMEOUT,

    // validation error
    DIMENSION_NOT_EQUAL = 2000,
    VECTOR_IS_EMPTY,
    JSON_PARSE_ERROR,
    DATA_UNMATCH_SCHEMA,  // this error code is to determine whether to update collection schema cache
};

/**
 * @brief Status code and message returned by SDK interface.
 */
class Status {
 public:
    /**
     * @brief Constructor of Status
     */
    Status(StatusCode code, std::string msg);
    Status(StatusCode code, std::string msg, int32_t rpc_err_code, int32_t server_err_code, int32_t legacy_server_code);
    Status();

    /**
     * @brief A success status
     */
    static Status
    OK();

    /**
     * @brief Indicate the status is ok
     */
    bool
    IsOk() const;

    /**
     * @brief Return the status code(general client-side error code)
     */
    StatusCode
    Code() const;

    /**
     * @brief Return the error message
     */
    const std::string&
    Message() const;

    /**
     * @brief The error code from gRPC lib, which are listed here:
     *      https://grpc.github.io/grpc/cpp/md_doc_statuscodes.html
     */
    int32_t
    RpcErrCode() const;

    /**
     * @brief The server-side error code of milvus v2.4 and later versions
     */
    int32_t
    ServerCode() const;

    /**
     * @brief The legacy server-side error code of milvus v2.2/v2.3
     */
    int32_t
    LegacyServerCode() const;

 private:
    StatusCode code_{StatusCode::OK};
    std::string msg_{"OK"};

    int32_t rpc_err_code_{0};        // rpc error code
    int32_t server_err_code_{0};     // v2.3+ server returns this code
    int32_t legacy_server_code_{0};  // to compatible with v2.2.x server
};

}  // namespace milvus
