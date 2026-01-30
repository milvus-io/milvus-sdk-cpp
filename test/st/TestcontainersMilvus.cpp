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

#include "TestcontainersMilvus.h"

#include <cstdlib>
#include <iostream>
#include <string>

#ifdef MILVUS_WITH_TESTCONTAINERS
#include <testcontainers-c.h>
#endif

namespace milvus {
namespace test {

#ifdef MILVUS_WITH_TESTCONTAINERS
namespace {
constexpr const char* kMilvusImage = "milvusdb/milvus:v2.4.23";
constexpr int kMilvusGrpcPort = 19530;
constexpr int kMilvusHttpPort = 9091;

void
SetEnvVar(const char* key, const std::string& value) {
#ifdef _WIN32
    _putenv_s(key, value.c_str());
#else
    setenv(key, value.c_str(), 1);
#endif
}

bool
ParseHostPortFromUri(const std::string& uri, std::string& host, std::uint16_t& port) {
    if (uri.empty()) {
        return false;
    }
    std::string work = uri;
    const auto scheme_pos = work.find("://");
    if (scheme_pos != std::string::npos) {
        work = work.substr(scheme_pos + 3);
    }
    const auto slash_pos = work.find('/');
    if (slash_pos != std::string::npos) {
        work = work.substr(0, slash_pos);
    }
    const auto colon_pos = work.rfind(':');
    if (colon_pos == std::string::npos) {
        return false;
    }
    host = work.substr(0, colon_pos);
    if (host.empty()) {
        return false;
    }
    const std::string port_text = work.substr(colon_pos + 1);
    if (port_text.empty()) {
        return false;
    }
    char* end = nullptr;
    const auto parsed = std::strtoul(port_text.c_str(), &end, 10);
    if (end == port_text.c_str() || parsed > 65535) {
        return false;
    }
    port = static_cast<std::uint16_t>(parsed);
    return true;
}
}  // namespace
#endif

MilvusTestcontainersEnvironment::MilvusTestcontainersEnvironment(bool enabled) : enabled_(enabled) {
}

void
MilvusTestcontainersEnvironment::SetUp() {
#ifdef MILVUS_WITH_TESTCONTAINERS
    if (!enabled_) {
        return;
    }

    const int request_id = static_cast<int>(tc_new_container_request(kMilvusImage));
    if (request_id < 0) {
        std::cerr << "Failed to create Milvus container request" << std::endl;
        enabled_ = false;
        return;
    }

    tc_with_exposed_tcp_port(request_id, kMilvusGrpcPort);
    tc_with_exposed_tcp_port(request_id, kMilvusHttpPort);
    tc_with_wait_for_http(request_id, kMilvusHttpPort, "/healthz");

    const auto run_result = tc_run_container(request_id);
    if (!run_result.r1) {
        std::cerr << "Failed to start Milvus container" << std::endl;
        enabled_ = false;
        return;
    }
    container_id_ = static_cast<int>(run_result.r0);

    const auto uri_result = tc_get_uri(container_id_, kMilvusGrpcPort);
    std::string uri;
    if (uri_result.r0.p != nullptr && uri_result.r0.n > 0) {
        uri.assign(uri_result.r0.p, static_cast<size_t>(uri_result.r0.n));
    }
    if (uri.empty()) {
        std::cerr << "Failed to get Milvus container URI" << std::endl;
        enabled_ = false;
        return;
    }

    std::string parsed_host;
    std::uint16_t parsed_port = kMilvusGrpcPort;
    if (!ParseHostPortFromUri(uri, parsed_host, parsed_port)) {
        std::cerr << "Failed to parse Milvus container URI: " << uri << std::endl;
        enabled_ = false;
        return;
    }

    host_ = parsed_host;
    port_ = parsed_port;
    SetEnvVar("MILVUS_TEST_HOST", host_);
    SetEnvVar("MILVUS_TEST_PORT", std::to_string(port_));

    std::cout << "Milvus test container started at " << host_ << ":" << port_ << std::endl;
#else
    (void)enabled_;
#endif
}

void
MilvusTestcontainersEnvironment::TearDown() {
#ifdef MILVUS_WITH_TESTCONTAINERS
    if (!enabled_) {
        return;
    }
    if (container_id_ < 0) {
        return;
    }
    char* error = tc_terminate_container(container_id_);
    if (error != nullptr && error[0] != '\0') {
        std::cerr << "Failed to terminate Milvus container: " << error << std::endl;
    }
#endif
}

bool
MilvusTestcontainersEnvironment::Enabled() const {
    return enabled_;
}

}  // namespace test
}  // namespace milvus
