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

#include "milvus/MilvusClient.h"

namespace milvus {
namespace test {

class MilvusServerTest : public ::testing::Test {
 protected:
    std::shared_ptr<milvus::MilvusClient> client_{nullptr};

    void
    SetUp() override {
        milvus::ConnectParam connect_param{"localhost", 19530};
        client_ = milvus::MilvusClient::Create();
        client_->Connect(connect_param);
    }

    void
    TearDown() override {
        client_->Disconnect();
    }
};

template <typename T>
class MilvusServerTestWithParam : public ::testing::TestWithParam<T> {
 protected:
    std::shared_ptr<milvus::MilvusClient> client_{nullptr};

    void
    SetUp() override {
        milvus::ConnectParam connect_param{"localhost", 19530};
        client_ = milvus::MilvusClient::Create();
        client_->Connect(connect_param);
    }

    void
    TearDown() override {
        client_->Disconnect();
    }
};

std::string
RanName(const std::string& prefix);

}  // namespace test
}  // namespace milvus
