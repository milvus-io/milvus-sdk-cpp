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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <thread>

#include "PythonMilvusServer.h"
#include "milvus/MilvusClient.h"
#include "milvus/Status.h"

namespace milvus {
namespace test {

inline void
waitMilvusServerReady(const PythonMilvusServer& server) {
    int max_retry = 60, retry = 0;
    bool has;

    auto client = milvus::MilvusClient::Create();
    auto param = server.TestClientParam();
    client->Connect(*param);
    auto status = client->HasCollection("no_such", has);

    while (!status.IsOk() && retry++ < max_retry) {
        std::this_thread::sleep_for(std::chrono::seconds{5});
        client = milvus::MilvusClient::Create();
        client->Connect(*param);
        status = client->HasCollection("no_such", has);
        std::cout << "Wait milvus start done, try: " << retry << ", status: " << status.Message() << std::endl;
    }
    std::cout << "Wait milvus start done, status: " << status.Message() << std::endl;
}

class MilvusServerTest : public ::testing::Test {
 protected:
    PythonMilvusServer server_{};
    std::shared_ptr<milvus::MilvusClient> client_{nullptr};

    void
    SetUp() override {
        server_.Start();
        client_ = milvus::MilvusClient::Create();
        waitMilvusServerReady(server_);
    }

    void
    TearDown() override {
    }
};

template <typename T>
class MilvusServerTestWithParam : public ::testing::TestWithParam<T> {
 protected:
    PythonMilvusServer server_{};
    std::shared_ptr<milvus::MilvusClient> client_{nullptr};

    void
    SetUp() override {
        server_.Start();
        client_ = milvus::MilvusClient::Create();
        waitMilvusServerReady(server_);
    }

    void
    TearDown() override {
    }
};
}  // namespace test
}  // namespace milvus
