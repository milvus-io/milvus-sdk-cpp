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

#include "PythonMilvusServer.h"

#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
// using 2.2.x latest
const char* kPythonMilvusServerVersion = "python-milvus-server~=2.2.0";
}  // namespace

PythonMilvusServer::~PythonMilvusServer() noexcept {
    Stop();
}

bool
PythonMilvusServer::AuthorizationEnabled() const {
    return authorization_enabled_;
}

void
PythonMilvusServer::SetAuthorizationEnabled(bool val) {
    authorization_enabled_ = val;
}

void
PythonMilvusServer::Start() {
    // install command
    std::string cmd = std::string("pip3 install -U ") + kPythonMilvusServerVersion;
    auto ret = system(cmd.c_str());
    if (ret != 0) {
        auto error = cmd + ", failed.";
        throw std::runtime_error(error);
    }
    // clear data before start
    cmd = "rm -fr " + base_dir_;
    system(cmd.c_str());
    thread_ = std::thread([this]() { this->run(); });
}

void
PythonMilvusServer::Stop() {
    if (pid_) {
        kill(pid_, SIGINT);
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

void
PythonMilvusServer::run() {
    // configs
    std::string auth_config = "authorization_enabled=false";
    std::string listen_config = "listen_port=19530";
    if (authorization_enabled_) {
        auth_config = "authorization_enabled=true";
    }

    const char* milvus_server = "milvus-server";

    pid_ = fork();
    if (pid_ == 0) {
        auto ret = execl("/usr/bin/env", "env", milvus_server, "--set", auth_config.c_str(), "--set",
                         listen_config.c_str(), "--data", base_dir_.c_str(), nullptr);
        exit(ret);
    }
    wait(&status_);
}

uint16_t
PythonMilvusServer::ListenPort() const {
    return 19530;
}