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

#include "milvus/Export.h"
#include "milvus/types/TelemetryConfig.h"

namespace milvus {

/**
 * @brief Connection parameters. Used by MilvusClient::Connect()
 * @par Example
 * @code
 * milvus::ConnectParam param{"localhost", 19530};
 * param.SetToken("root:Milvus");
 * auto client = milvus::MilvusClientV2::Create();
 * auto status = client->Connect(param);
 * @endcode
 */
class MILVUS_SDK_API ConnectParam {
 public:
    /**
     * @brief Constructor
     */
    ConnectParam() = default;

    ConnectParam&
    operator=(const ConnectParam&);

    /**
     * @brief Constructor
     * @param uri URI for connecting to Milvus; it can be a cloud instance endpoint or an address such as
     * "http://xx.xx.xx.xx:19530".
     */
    explicit ConnectParam(const std::string& uri);

    /**
     * @brief Constructor
     * @param uri URI for connecting to Milvus; it can be a cloud instance endpoint or an address such as
     * "http://xx.xx.xx.xx:19530".
     * @param token Authorization value for connecting to Milvus, in the format "[user]:[password]" or as a cloud
     * instance token.
     */
    ConnectParam(const std::string& uri, const std::string& token);

    /**
     * @brief Constructor
     * @deprecated host/port is replaced by uri
     * @param [in] host the host.
     * @param [in] port the port.
     */
    ConnectParam(std::string host, uint16_t port);

    /**
     * @brief Constructor
     * @deprecated host/port is replaced by uri
     * @param [in] host the host.
     * @param [in] port the port.
     * @param [in] token the token.
     */
    ConnectParam(std::string host, uint16_t port, const std::string& token);

    /**
     * @brief Constructor
     * @deprecated host/port is replaced by uri
     * @param [in] host the host.
     * @param [in] port the port.
     * @param [in] username the username.
     * @param [in] password the password.
     */
    ConnectParam(std::string host, uint16_t port, std::string username, std::string password);

    /**
     * @brief IP address of the Milvus proxy.
     * @return the host.
     */
    std::string
    Host() const;

    /**
     * @brief Port of the Milvus proxy.
     * @return the port.
     */
    uint16_t
    Port() const;

    /**
     * @brief URI for connecting to Milvus.
     * @return the URI.
     */
    std::string
    Uri() const;

    /**
     * @brief Set the URI for connecting to Milvus.
     * @param [in] uri the URI.
     */
    void
    SetUri(const std::string& uri);

    /**
     * @brief Set the URI for connecting to Milvus.
     * @param [in] uri the URI.
     */
    ConnectParam&
    WithUri(const std::string& uri);

    /**
     * @brief Token for connecting to Milvus.
     * @return the token.
     */
    const std::string&
    Token() const;

    /**
     * @brief Set the token for connecting to Milvus.
     * Note: calling this method resets the username and password.
     * @param [in] token the token.
     */
    void
    SetToken(const std::string& token);

    /**
     * @brief Set the token for connecting to Milvus.
     * Note: calling this method resets the username and password.
     * @param [in] token the token.
     */
    ConnectParam&
    WithToken(const std::string& token);

    /**
     * @brief Authorization header value for connecting to Milvus.
     * Authorizations() = base64('username:password').
     * @return the authorizations.
     */
    const std::string&
    Authorizations() const;

    /**
     * @brief Set the username and password used to connect to Milvus.
     * Note: calling this method resets the token.
     * @param [in] username the username.
     * @param [in] password the password.
     */
    void
    SetAuthorizations(std::string username, std::string password);

    /**
     * @brief Set the username and password used to connect to Milvus.
     * Note: calling this method resets the token.
     * @param [in] username the username.
     * @param [in] password the password.
     */
    ConnectParam&
    WithAuthorizations(std::string username, std::string password);

    /**
     * @brief Connect timeout in milliseconds.
     *
     * @return the connect timeout.
     */
    uint64_t
    ConnectTimeout() const;

    /**
     * @brief Set connect timeout in milliseconds. It is the timeout value to wait grpc channel to ready.
     *
     * @param [in] connect_timeout_ms the connect timeout ms.
     */
    void
    SetConnectTimeout(uint64_t connect_timeout_ms);

    /**
     * @brief Set connect timeout in milliseconds. It is the timeout value to wait grpc channel to ready.
     * @param [in] connect_timeout_ms the connect timeout ms.
     */
    ConnectParam&
    WithConnectTimeout(uint64_t connect_timeout_ms);

    /**
     * @brief Get the keepalive interval in milliseconds.
     *
     * Read the gRPC documentation for more information:
     * https://github.com/grpc/grpc/blob/master/doc/keepalive.md
     * @return the keepalive time ms.
     */
    uint64_t
    KeepaliveTimeMs() const;

    /**
     * @brief Set keepalive time value in milliseconds.
     *
     * @param [in] keepalive_time_ms the keepalive time ms.
     */
    void
    SetKeepaliveTimeMs(uint64_t keepalive_time_ms);

    /**
     * @brief Set keepalive time value in milliseconds.
     * @param [in] keepalive_time_ms the keepalive time ms.
     */
    ConnectParam&
    WithKeepaliveTimeMs(uint64_t keepalive_time_ms);

    /**
     * @brief Get keepalive timeout value milliseconds.
     *
     * @return the keepalive timeout ms.
     */
    uint64_t
    KeepaliveTimeoutMs() const;

    /**
     * @brief Set keepalive timeout value in milliseconds.
     *
     * @param [in] keepalive_timeout_ms the keepalive timeout ms.
     */
    void
    SetKeepaliveTimeoutMs(uint64_t keepalive_timeout_ms);

    /**
     * @brief Set keepalive timeout value in milliseconds.
     * @param [in] keepalive_timeout_ms the keepalive timeout ms.
     */
    ConnectParam&
    WithKeepaliveTimeoutMs(uint64_t keepalive_timeout_ms);

    /**
     * @brief Get keepalive without calls value.
     *
     * @return the keepalive without calls.
     */
    bool
    KeepaliveWithoutCalls() const;

    /**
     * @brief Set keepalive without calls or not.
     *
     * @param [in] keepalive_without_calls the keepalive without calls.
     */
    void
    SetKeepaliveWithoutCalls(bool keepalive_without_calls);

    /**
     * @brief Set keepalive without calls or not.
     * @param [in] keepalive_without_calls the keepalive without calls.
     */
    ConnectParam&
    WithKeepaliveWithoutCalls(bool keepalive_without_calls);

    /**
     * @brief Get deadline value of rpc call in milliseconds.
     *
     * @return the RPC deadline ms.
     */
    uint64_t
    RpcDeadlineMs() const;

    /**
     * @brief Set deadline value of rpc call in milliseconds.
     *
     * @param [in] rpc_deadline_ms the RPC deadline ms.
     */
    void
    SetRpcDeadlineMs(uint64_t rpc_deadline_ms);

    /**
     * @brief Set deadline value of rpc call in milliseconds.
     * @param [in] rpc_deadline_ms the RPC deadline ms.
     */
    ConnectParam&
    WithRpcDeadlineMs(uint64_t rpc_deadline_ms);

    /**
     * @brief With ssl
     * @return the with tls.
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
     * @param [in] server_name the server name.
     * @param [in] ca_cert the ca cert.
     */
    ConnectParam&
    WithTls(const std::string& server_name, const std::string& ca_cert);

    /**
     * @brief Enable ssl
     * @param [in] server_name the server name.
     * @param [in] ca_cert the ca cert.
     */
    void
    EnableTls(const std::string& server_name, const std::string& ca_cert);

    /**
     * @brief With ssl and provides certificates
     * @return the with tls.
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
     * @return the tls enabled.
     */
    bool
    TlsEnabled() const;

    /**
     * @brief ServerName tls hostname
     * @return the server name.
     */
    const std::string&
    ServerName() const;

    /**
     * @brief Cert tls cert file
     * @return the cert.
     */
    const std::string&
    Cert() const;

    /**
     * @brief Key tls key file
     * @return the key.
     */
    const std::string&
    Key() const;

    /**
     * @brief CaCert tls ca cert file
     * @return the ca cert.
     */
    const std::string&
    CaCert() const;

    /**
     * @brief Return user name
     * @return the username.
     */
    const std::string&
    Username() const;

    /**
     * @brief Return the current used database name
     * @return the DB name.
     */
    std::string
    DbName() const;

    /**
     * @brief Set the current used database name
     * @param [in] db_name the DB name.
     */
    void
    SetDbName(const std::string& db_name);

    /**
     * @brief Set the current used database name
     * @param [in] db_name the DB name.
     */
    ConnectParam&
    WithDbName(const std::string& db_name);

    /**
     * @brief Get the telemetry configuration.
     * @return the telemetry.
     */
    const TelemetryConfig&
    Telemetry() const;

    /**
     * @brief Set the telemetry configuration.
     *
     * @param [in] config
     */
    void
    SetTelemetryConfig(const TelemetryConfig& config);

    /**
     * @brief Set the telemetry configuration.
     *
     * @param [in] config
     */
    ConnectParam&
    WithTelemetryConfig(const TelemetryConfig& config);

 private:
    std::string uri_ = "http://localhost:19530";

    uint64_t connect_timeout_ms_ = 10000;   // the same with pymilvus
    uint64_t keepalive_time_ms_ = 10000;    // Send keepalive pings every 10 seconds
    uint64_t keepalive_timeout_ms_ = 5000;  // Keepalive ping timeout after 5 seconds
    bool keepalive_without_calls_ = true;   // Allow keepalive pings when there are no gRPC calls
    uint64_t rpc_deadline_ms_ = 0;          // the same with java sdk

    bool tls_{false};
    std::string server_name_;
    std::string cert_;
    std::string key_;
    std::string ca_cert_;

    std::string authorizations_;
    std::string username_;
    std::string token_;
    std::string db_name_;
    TelemetryConfig telemetry_config_;
};

}  // namespace milvus
