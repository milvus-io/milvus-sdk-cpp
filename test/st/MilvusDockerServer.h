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

#include <map>
#include <memory>
#include <stdexcept>
#include <string>

class MilvusDockerServer {
 public:
    class TimeoutError : public std::runtime_error {
     public:
        TimeoutError(const std::string& msg) : std::runtime_error(msg) {
        }
    };

    class Process {
        std::string command_;
        int timeout_;
        std::string output_;

     public:
        Process(const std::string& command, int timeout);
        int
        operator()();
        const std::string&
        Output() const;
    };

    class DockerContainer {
        std::string container_name_;
        std::string image_name_;
        std::string image_tag_;
        int port_;
        int host_port_;
        std::string command_;
        std::map<std::string, std::string> envs_;

     public:
        DockerContainer(const std::string& container_name, const std::string& image_name, const std::string& tag,
                        int port, int host_port, const std::string& command,
                        const std::map<std::string, std::string>& envs);

        ~DockerContainer();

        void
        Start() const;

        void
        Stop() const;

        std::string
        Status() const;

        std::string
        Address() const;
    };

    MilvusDockerServer() = default;

    void
    Start();

    void
    Stop();

    void
    Reset();

    uint16_t
    ListenPort() const;

 private:
    std::shared_ptr<DockerContainer> etcd_container_{nullptr};
    std::shared_ptr<DockerContainer> minio_container_{nullptr};
    std::shared_ptr<DockerContainer> milvus_container_{nullptr};
};
