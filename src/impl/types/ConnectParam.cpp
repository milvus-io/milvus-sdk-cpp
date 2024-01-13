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

#include "../TypeUtils.h"

namespace milvus {

ConnectParam::ConnectParam(std::string host, uint16_t port) : host_(std::move(host)), port_(port) {
}

ConnectParam::ConnectParam(std::string host, uint16_t port, std::string username, std::string password)
    : host_(std::move(host)), port_(port) {
    SetAuthorizations(std::move(username), std::move(password));
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
}

uint32_t
ConnectParam::ConnectTimeout() const {
    return connect_timeout_;
}

void
ConnectParam::SetConnectTimeout(uint32_t timeout) {
    connect_timeout_ = timeout;
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

}  // namespace milvus
