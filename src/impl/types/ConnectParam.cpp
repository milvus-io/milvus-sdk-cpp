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

#include "milvus/types/ConnectParam.h"

#include <string>

#include "../utils/TypeUtils.h"

namespace milvus {

ConnectParam::ConnectParam(std::string host, uint16_t port) : host_(std::move(host)), port_(port) {
}

ConnectParam::ConnectParam(std::string host, uint16_t port, const std::string& token)
    : host_(std::move(host)), port_(port) {
    SetToken(token);
}

ConnectParam::ConnectParam(std::string host, uint16_t port, std::string username, std::string password)
    : host_(std::move(host)), port_(port) {
    SetAuthorizations(std::move(username), std::move(password));
}

ConnectParam&
ConnectParam::operator=(const ConnectParam& other) {
    if (this != &other) {
        host_ = other.host_;
        port_ = other.port_;

        connect_timeout_ms_ = other.connect_timeout_ms_;
        keepalive_time_ms_ = other.keepalive_time_ms_;
        keepalive_timeout_ms_ = other.keepalive_timeout_ms_;
        keepalive_without_calls_ = other.keepalive_without_calls_;
        rpc_deadline_ms_ = other.rpc_deadline_ms_;

        tls_ = other.tls_;
        server_name_ = other.server_name_;
        cert_ = other.cert_;
        key_ = other.key_;
        ca_cert_ = other.ca_cert_;

        authorizations_ = other.authorizations_;
        username_ = other.username_;
        db_name_ = other.db_name_;
    }
    return *this;
}

const std::string&
ConnectParam::Host() const {
    return host_;
}

uint16_t
ConnectParam::Port() const {
    return port_;
}

std::string
ConnectParam::Uri() const {
    return host_ + ":" + std::to_string(port_);
}

const std::string&
ConnectParam::Authorizations() const {
    return authorizations_;
}

void
ConnectParam::SetAuthorizations(std::string username, std::string password) {
    authorizations_ = milvus::Base64Encode(std::move(username) + ':' + std::move(password));
    username_ = username;
}

ConnectParam&
ConnectParam::WithAuthorizations(std::string username, std::string password) {
    SetAuthorizations(username, password);
    return *this;
}

uint64_t
ConnectParam::ConnectTimeout() const {
    return connect_timeout_ms_;
}

void
ConnectParam::SetConnectTimeout(uint64_t connect_timeout_ms) {
    connect_timeout_ms_ = connect_timeout_ms;
}

ConnectParam&
ConnectParam::WithConnectTimeout(uint64_t connect_timeout_ms) {
    SetConnectTimeout(connect_timeout_ms);
    return *this;
}

uint64_t
ConnectParam::KeepaliveTimeMs() const {
    return keepalive_time_ms_;
}

void
ConnectParam::SetKeepaliveTimeMs(uint64_t keepalive_time_ms) {
    keepalive_time_ms_ = keepalive_time_ms;
}

ConnectParam&
ConnectParam::WithKeepaliveTimeMs(uint64_t keepalive_time_ms) {
    SetKeepaliveTimeMs(keepalive_time_ms);
    return *this;
}

uint64_t
ConnectParam::KeepaliveTimeoutMs() const {
    return keepalive_timeout_ms_;
}

void
ConnectParam::SetKeepaliveTimeoutMs(uint64_t keepalive_timeout_ms) {
    keepalive_timeout_ms_ = keepalive_timeout_ms;
}

ConnectParam&
ConnectParam::WithKeepaliveTimeoutMs(uint64_t keepalive_timeout_ms) {
    SetKeepaliveTimeoutMs(keepalive_timeout_ms);
    return *this;
}

bool
ConnectParam::KeepaliveWithoutCalls() const {
    return keepalive_without_calls_;
}

void
ConnectParam::SetKeepaliveWithoutCalls(bool keepalive_without_calls) {
    keepalive_without_calls_ = keepalive_without_calls;
}

ConnectParam&
ConnectParam::WithKeepaliveWithoutCalls(bool keepalive_without_calls) {
    SetKeepaliveWithoutCalls(keepalive_without_calls);
    return *this;
}

uint64_t
ConnectParam::RpcDeadlineMs() const {
    return rpc_deadline_ms_;
}

void
ConnectParam::SetRpcDeadlineMs(uint64_t rpc_deadline_ms) {
    rpc_deadline_ms_ = rpc_deadline_ms;
}

ConnectParam&
ConnectParam::WithRpcDeadlineMs(uint64_t rpc_deadline_ms) {
    SetRpcDeadlineMs(rpc_deadline_ms);
    return *this;
}

ConnectParam&
ConnectParam::WithTls() {
    EnableTls();
    return *this;
}

void
ConnectParam::EnableTls() {
    EnableTls("", "", "", "");
}

ConnectParam&
ConnectParam::WithTls(const std::string& server_name, const std::string& ca_cert) {
    EnableTls(server_name, ca_cert);
    return *this;
}

void
ConnectParam::EnableTls(const std::string& server_name, const std::string& ca_cert) {
    EnableTls(server_name, "", "", ca_cert);
}

ConnectParam&
ConnectParam::WithTls(const std::string& server_name, const std::string& cert, const std::string& key,
                      const std::string& ca_cert) {
    EnableTls(server_name, cert, key, ca_cert);
    return *this;
}

void
ConnectParam::EnableTls(const std::string& server_name, const std::string& cert, const std::string& key,
                        const std::string& ca_cert) {
    tls_ = true;
    server_name_ = server_name;
    cert_ = cert;
    key_ = key;
    ca_cert_ = ca_cert;
}

void
ConnectParam::DisableTls() {
    tls_ = false;
    server_name_.clear();
    cert_.clear();
    key_.clear();
    ca_cert_.clear();
}

bool
ConnectParam::TlsEnabled() const {
    return tls_;
}

const std::string&
ConnectParam::ServerName() const {
    return server_name_;
}

const std::string&
ConnectParam::Cert() const {
    return cert_;
}

const std::string&
ConnectParam::Key() const {
    return key_;
}

const std::string&
ConnectParam::CaCert() const {
    return ca_cert_;
}

const std::string&
ConnectParam::Username() const {
    return username_;
}

void
ConnectParam::SetToken(const std::string& token) {
    authorizations_ = milvus::Base64Encode(token);
    username_ = "";
}

ConnectParam&
ConnectParam::WithToken(const std::string& token) {
    SetToken(token);
    return *this;
}

const std::string&
ConnectParam::DbName() const {
    return db_name_;
}

void
ConnectParam::SetDbName(const std::string& db_name) {
    db_name_ = db_name;
}

ConnectParam&
ConnectParam::WithDbName(const std::string& db_name) {
    SetDbName(db_name);
    return *this;
}

}  // namespace milvus
