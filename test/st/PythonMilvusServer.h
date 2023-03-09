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

#include <memory>
#include <string>
#include <thread>

#include "milvus/types/ConnectParam.h"

namespace milvus {

namespace test {

class PythonMilvusServer {
    // enable auth
    bool authorization_enabled_{false};

    // if using tls
    int tls_mode_{0};
    std::string server_cert_;
    std::string server_key_;
    std::string ca_cert_;

    // base data dir
    std::string base_dir_{"/tmp/milvus_data"};

    std::thread thread_;

    int status_{0};
    int pid_{0};

    void
    run();

 public:
    ~PythonMilvusServer() noexcept;

    void
    SetAuthorizationEnabled(bool val);

    void
    SetTls(int mode, const std::string& server_cert, const std::string& server_key, const std::string& ca_cert);

    void
    Start();

    void
    Stop();

    uint16_t
    ListenPort() const;

    std::shared_ptr<milvus::ConnectParam>
    TestClientParam() const;
};

}  // namespace test
}  // namespace milvus