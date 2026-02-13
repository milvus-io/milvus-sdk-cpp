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

namespace milvus {

/**
 * @brief Connection parameters. Used by MilvusClient::Connect()
 */
class ConnectParam {
 public:
    ConnectParam() = default;

    ConnectParam&
    operator=(const ConnectParam&);

    /**
     * @brief Constructor
     * @param uri Uri for connecting to the milvus, can be an endpoint of cloud instance or address like
     * "http://xx.xx.xx.xx:19530".
     */
    explicit ConnectParam(const std::string& uri);

    /**
     * @brief Constructor
     * @param uri Uri for connecting to the milvus, can be an endpoint of cloud instance or address like
     * "http://xx.xx.xx.xx:19530".
     * @param token Authorizations header value for connecting to the milvus. In the format of "[user]:[password]" or
     * cloud instance token.
     */
    ConnectParam(const std::string& uri, const std::string& token);

    /**
     * @brief Constructor
     * @deprecated host/port is replaced by uri
     */
    ConnectParam(std::string host, uint16_t port);

    /**
     * @brief Constructor
     * @deprecated host/port is replaced by uri
     */
    ConnectParam(std::string host, uint16_t port, const std::string& token);

    /**
     * @brief Constructor
     * @deprecated host/port is replaced by uri
     */
    ConnectParam(std::string host, uint16_t port, std::string username, std::string password);

    /**
     * @brief IP of the milvus proxy.
     */
    std::string
    Host() const;

    /**
     * @brief Port of the milvus proxy.
     */
    uint16_t
    Port() const;

    /**
     * @brief Uri for connecting to the milvus.
     */
    std::string
    Uri() const;

    /**
     * @brief Set Uri for connecting to the milvus.
     */
    void
    SetUri(const std::string& uri);

    /**
     * @brief Set Uri for connecting to the milvus.
     */
    ConnectParam&
    WithUri(const std::string& uri);

    /**
     * @brief Token for connecting to the milvus.
     */
    const std::string&
    Token() const;

    /**
     * @brief Set token for connecting to the milvus.
     * Note: call this method will reset username/password.
     */
    void
    SetToken(const std::string& token);

    /**
     * @brief Set token for connecting to the milvus.
     * Note: call this method will reset username/password.
     */
    ConnectParam&
    WithToken(const std::string& token);

    /**
     * @brief Authorizations header value for connecting to the milvus.
     * Authorizations() = base64('username:password').
     */
    const std::string&
    Authorizations() const;

    /**
     * @brief SetAuthorizations set username and password for connecting to the milvus.
     * Note: call this method will reset the token.
     */
    void
    SetAuthorizations(std::string username, std::string password);

    /**
     * @brief SetAuthorizations set username and password for connecting to the milvus.
     * Note: call this method will reset the token.
     */
    ConnectParam&
    WithAuthorizations(std::string username, std::string password);

    /**
     * @brief Connect timeout in milliseconds.
     *
     */
    uint64_t
    ConnectTimeout() const;

    /**
     * @brief Set connect timeout in milliseconds. It is the timeout value to wait grpc channel to ready.
     *
     */
    void
    SetConnectTimeout(uint64_t connect_timeout_ms);

    /**
     * @brief Set connect timeout in milliseconds. It is the timeout value to wait grpc channel to ready.
     */
    ConnectParam&
    WithConnectTimeout(uint64_t connect_timeout_ms);

    /**
     * @brief Get keepalive time value milliseconds.
     *
     * Note: teke effect when is true
     * read the grpc doc for more info: https://github.com/grpc/grpc/blob/master/doc/keepalive.md
     */
    uint64_t
    KeepaliveTimeMs() const;

    /**
     * @brief Set keepalive time value in milliseconds.
     *
     */
    void
    SetKeepaliveTimeMs(uint64_t keepalive_time_ms);

    /**
     * @brief Set keepalive time value in milliseconds.
     */
    ConnectParam&
    WithKeepaliveTimeMs(uint64_t keepalive_time_ms);

    /**
     * @brief Get keepalive timeout value milliseconds.
     *
     */
    uint64_t
    KeepaliveTimeoutMs() const;

    /**
     * @brief Set keepalive timeout value in milliseconds.
     *
     */
    void
    SetKeepaliveTimeoutMs(uint64_t keepalive_timeout_ms);

    /**
     * @brief Set keepalive timeout value in milliseconds.
     */
    ConnectParam&
    WithKeepaliveTimeoutMs(uint64_t keepalive_timeout_ms);

    /**
     * @brief Get keepalive without calls value.
     *
     */
    bool
    KeepaliveWithoutCalls() const;

    /**
     * @brief Set keepalive without calls or not.
     *
     */
    void
    SetKeepaliveWithoutCalls(bool keepalive_without_calls);

    /**
     * @brief Set keepalive without calls or not.
     */
    ConnectParam&
    WithKeepaliveWithoutCalls(bool keepalive_without_calls);

    /**
     * @brief Get deadline value of rpc call in milliseconds.
     *
     */
    uint64_t
    RpcDeadlineMs() const;

    /**
     * @brief Set deadline value of rpc call in milliseconds.
     *
     */
    void
    SetRpcDeadlineMs(uint64_t rpc_deadline_ms);

    /**
     * @brief Set deadline value of rpc call in milliseconds.
     */
    ConnectParam&
    WithRpcDeadlineMs(uint64_t rpc_deadline_ms);

    /**
     * @brief With ssl
     */
    ConnectParam&
    WithTls();

    /**
     * @brief Enable ssl
     */
    void
    EnableTls();

    /**
     * @brief With ssl
     */
    ConnectParam&
    WithTls(const std::string& server_name, const std::string& ca_cert);

    /**
     * @brief Enable ssl
     */
    void
    EnableTls(const std::string& server_name, const std::string& ca_cert);

    /**
     * @brief With ssl and provides certificates
     */
    ConnectParam&
    WithTls(const std::string& server_name, const std::string& cert, const std::string& key,
            const std::string& ca_cert);

    /**
     * @brief Enable ssl and provides certificates
     */
    void
    EnableTls(const std::string& server_name, const std::string& cert, const std::string& key,
              const std::string& ca_cert);

    /**
     * @brief Disable ssl
     */
    void
    DisableTls();

    /**
     * @brief TlsEnabled
     */
    bool
    TlsEnabled() const;

    /**
     * @brief ServerName tls hostname
     */
    const std::string&
    ServerName() const;

    /**
     * @brief Cert tls cert file
     */
    const std::string&
    Cert() const;

    /**
     * @brief Key tls key file
     */
    const std::string&
    Key() const;

    /**
     * @brief CaCert tls ca cert file
     */
    const std::string&
    CaCert() const;

    /**
     * @brief Return user name
     */
    const std::string&
    Username() const;

    /**
     * @brief Return the current used database name
     */
    std::string
    DbName() const;

    /**
     * @brief Set the current used database name
     */
    void
    SetDbName(const std::string& db_name);

    /**
     * @brief Set the current used database name
     */
    ConnectParam&
    WithDbName(const std::string& db_name);

 private:
    std::string uri_ = "http://localhost:19530";

    uint64_t connect_timeout_ms_ = 10000;    // the same with pymilvus
    uint64_t keepalive_time_ms_ = 10000;     // Send keepalive pings every 10 seconds
    uint64_t keepalive_timeout_ms_ = 5000;   // Keepalive ping timeout after 5 seconds
    bool keepalive_without_calls_ = true;    // Allow keepalive pings when there are no gRPC calls
    uint64_t rpc_deadline_ms_ = 0;           // the same with java sdk

    bool tls_{false};
    std::string server_name_;
    std::string cert_;
    std::string key_;
    std::string ca_cert_;

    std::string authorizations_;
    std::string username_;
    std::string token_;
    std::string db_name_;
};

}  // namespace milvus
