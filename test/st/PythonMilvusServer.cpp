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

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace milvus {

namespace test {

static void
generate_certificates() {
    std::system("mkdir -p certs");
    std::system("openssl genrsa -out certs/ca.key 2048");
    std::system(
        "openssl req -new"
        " -key certs/ca.key"
        " -subj /C=CN/ST=Zhejiang/L=Hangzhou/O=Milvus/OU=CppSdk/CN=ca.test.com"
        " -out certs/ca.csr");
    std::system(
        "openssl x509 -req"
        " -days 365"
        " -in certs/ca.csr"
        " -signkey certs/ca.key"
        " -out certs/ca.crt");
    for (const auto& name : {"server", "client"}) {
        std::system((std::string("openssl genrsa -out certs/") + name + ".key 2048").c_str());
        std::system((std::string("openssl req -new -key certs/") + name +
                     ".key"
                     " -subj /C=CN/ST=Zhejiang/L=Hangzhou/O=Milvus/OU=CppSdk/CN=" +
                     name +
                     ".test.com"
                     " -out certs/" +
                     name + ".csr")
                        .c_str());
        std::system((std::string("openssl x509 -req -days 365 -in certs/") + name +
                     ".csr"
                     " -CA certs/ca.crt -CAkey certs/ca.key -CAcreateserial"
                     " -out certs/" +
                     name + ".crt")
                        .c_str());
    }
    std::system("echo generate certifications");
}

// using 2.3.x latest
const char* kPythonMilvusServerVersion = "milvus~=2.3.0";

PythonMilvusServer::~PythonMilvusServer() noexcept {
}

void
PythonMilvusServer::SetAuthorizationEnabled(bool val) {
    authorization_enabled_ = val;
}

void
PythonMilvusServer::SetTls(int mode, const std::string& server_cert, const std::string& server_key,
                           const std::string& ca_cert) {
    tls_mode_ = mode;
    server_cert_ = server_cert;
    server_key_ = server_key;
    ca_cert_ = ca_cert;
}

void
PythonMilvusServer::Start() {
    if (Started()) {
        return;
    }
    // install command
    std::string cmd = std::string("pip3 install ") + kPythonMilvusServerVersion;
    auto ret = system(cmd.c_str());
    if (ret != 0) {
        auto error = cmd + ", failed.";
        throw std::runtime_error(error);
    }
    // clear data before start
    cmd = "rm -fr " + base_dir_;
    system(cmd.c_str());
    thread_ = std::thread([this]() { this->run(); });
    started_ = true;
}

void
PythonMilvusServer::Stop() {
    if (!Started()) {
        return;
    }
    if (pid_) {
        kill(pid_, SIGINT);
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    // sleep 5s to wait for release port
    std::this_thread::sleep_for(std::chrono::seconds(5));
    started_ = false;
}

void
PythonMilvusServer::run() {
    // configs
    std::string cmd = "milvus-server --data " + base_dir_;
    if (authorization_enabled_) {
        cmd += " --authorization-enabled true";
    }

    if (tls_mode_ != 0) {
        generate_certificates();

        cmd += " --tls-mode " + std::to_string(tls_mode_);
        cmd += " --server-pem-path " + server_cert_;
        cmd += " --server-key-path " + server_key_;
        cmd += " --ca-pem-path " + ca_cert_;
    }

    pid_ = fork();
    if (pid_ == 0) {
        auto ret = execl("/bin/bash", "bash", "-c", cmd.c_str(), nullptr);
        exit(ret);
    }
    wait(&status_);
}

uint16_t
PythonMilvusServer::ListenPort() const {
    return 19530;
}

std::shared_ptr<milvus::ConnectParam>
PythonMilvusServer::TestClientParam() const {
    auto param = std::make_shared<ConnectParam>("127.0.0.1", ListenPort());
    if (authorization_enabled_) {
        // root enabled by default.
        param->SetAuthorizations("root", "Milvus");
    }
    if (tls_mode_ > 0) {
        std::array<char, 256> path;
        getcwd(path.data(), path.size());
        std::string pwd = path.data();
        std::string server = "server.test.com";

        if (tls_mode_ == 1) {
            param->EnableTls(server, pwd + "/certs/ca.crt");
        } else if (tls_mode_ == 2) {
            param->EnableTls(server, pwd + "/certs/client.crt", pwd + "/certs/client.key", pwd + "/certs/ca.crt");
        }
    }
    return param;
}
}  // namespace test
}  // namespace milvus
