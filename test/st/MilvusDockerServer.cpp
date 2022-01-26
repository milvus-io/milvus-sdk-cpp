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

#include "MilvusDockerServer.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <thread>

#include "milvus/MilvusClient.h"

namespace {
constexpr uint16_t milvus_listen_port{19530};
constexpr char etcd_image_repo[] = "quay.io/coreos/etcd";
constexpr char etcd_image_tag[] = "v3.5.0";
constexpr char etcd_start_cmd[] =
    "etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd";

constexpr char minio_image_repo[] = "minio/minio";
constexpr char minio_image_tag[] = "RELEASE.2020-12-03T00-03-10Z";
constexpr char minio_start_cmd[] = "minio server /minio_data";

constexpr char milvus_image_repo[] = "milvusdb/milvus";
constexpr char milvus_image_tag[] = "v2.0.0";
constexpr char milvus_start_cmd[] = "milvus run standalone";
}  // namespace

MilvusDockerServer::Process::Process(const std::string& command, int timeout) : command_(command), timeout_(timeout) {
}

int
MilvusDockerServer::Process::operator()() {
    fd_set set;
    std::string command = "timeout " + std::to_string(timeout_) + " " + command_;
    auto proc = popen(command.c_str(), "r");
    std::stringstream stdout_buff{};
    std::array<char, 256> buff{};

    auto read_bytes = fread(buff.data(), sizeof(char), buff.size(), proc);
    while (read_bytes > 0) {
        stdout_buff << std::string(buff.data(), read_bytes);
        read_bytes = fread(buff.data(), sizeof(char), buff.size(), proc);
    }
    output_ = stdout_buff.str();
    output_.erase(0, output_.find_first_not_of("\r\n "));
    output_.erase(output_.find_last_not_of("\r\n ") + 1);
    return pclose(proc);
}

const std::string&
MilvusDockerServer::Process::Output() const {
    return output_;
}

MilvusDockerServer::DockerContainer::DockerContainer(const std::string& container_name, const std::string& image_name,
                                                     const std::string& tag, int port, int host_port,
                                                     const std::string& command,
                                                     const std::map<std::string, std::string>& envs)
    : container_name_(container_name),
      image_name_(image_name),
      image_tag_(tag),
      port_(port),
      host_port_(host_port),
      command_(command),
      envs_(envs) {
    try {
        Start();
    } catch (const std::runtime_error&) {
        Stop();
        Start();
    }
}

MilvusDockerServer::DockerContainer::~DockerContainer() {
    Stop();
}

void
MilvusDockerServer::DockerContainer::Start() const {
    std::stringstream start_command;
    start_command << "docker run --rm -d ";
    if (host_port_ > 0 && port_ > 0) {
        start_command << "--publish " << host_port_ << ":" << port_ << " ";
    }
    start_command << "--name " << container_name_ << " ";
    for (const auto& pair : envs_) {
        start_command << "-e " << pair.first << "=" << pair.second << " ";
    }
    start_command << image_name_ << ":" << image_tag_ << " " << command_;
    Process start_docker{start_command.str(), 120};
    if (start_docker() != 0) {
        throw std::runtime_error("Failed to start docker container: " + container_name_ + ", output" +
                                 start_docker.Output());
    }
}

void
MilvusDockerServer::DockerContainer::Stop() const {
    Process stop_docker{"docker stop " + container_name_, 10};
    stop_docker();
}

std::string
MilvusDockerServer::DockerContainer::Status() const {
    Process status_docker{"docker inspect " + container_name_ + " | jq -r '.[0].State.Status'", 10};
    status_docker();
    return status_docker.Output();
}

std::string
MilvusDockerServer::DockerContainer::Address() const {
    Process status_docker{
        " docker inspect " + container_name_ + " | jq -r '.[].NetworkSettings.Networks | to_entries[].value.IPAddress'",
        10};
    status_docker();
    return status_docker.Output();
}

void
MilvusDockerServer::Start() {
    etcd_container_ = std::make_shared<DockerContainer>(
        "test-milvus-etcd", etcd_image_repo, etcd_image_tag, 0, 0, etcd_start_cmd,
        std::map<std::string, std::string>{{"ETCD_AUTO_COMPACTION_MODE", "revision"},
                                           {"ETCD_AUTO_COMPACTION_RETENTION", "1000"},
                                           {"ETCD_QUOTA_BACKEND_BYTES", "4294967296"}});

    minio_container_ = std::make_shared<DockerContainer>(
        "test-milvus-minio", minio_image_repo, minio_image_tag, 0, 0, minio_start_cmd,
        std::map<std::string, std::string>{{"MINIO_ACCESS_KEY", "minioadmin"}, {"MINIO_SECRET_KEY", "minioadmin"}});

    milvus_container_ = std::make_shared<DockerContainer>(
        "test-milvus-milvus", milvus_image_repo, milvus_image_tag, milvus_listen_port, milvus_listen_port,
        milvus_start_cmd,
        std::map<std::string, std::string>{{"ETCD_ENDPOINTS", etcd_container_->Address() + ":2379"},
                                           {"MINIO_ADDRESS", minio_container_->Address() + ":9000"}});

    // wait connection ready in 10 seconds
    milvus::ConnectParam connect_param{"127.0.0.1", ListenPort()};
    bool has{false};
    auto dead_time = std::chrono::system_clock::now() + std::chrono::seconds{10};
    while (dead_time > std::chrono::system_clock::now()) {
        auto client = milvus::MilvusClient::Create();
        client->Connect(connect_param);
        auto status = client->HasCollection("not_exist_dummy_collection", has);
        if (status.IsOk()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
}

void
MilvusDockerServer::Stop() {
    milvus_container_ = nullptr;
    minio_container_ = nullptr;
    etcd_container_ = nullptr;
}

void
MilvusDockerServer::Reset() {
    Stop();
    Start();
}

uint16_t
MilvusDockerServer::ListenPort() const {
    return milvus_listen_port;
}