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

#include <gtest/gtest.h>

#include <random>

#include "MilvusServerTest.h"

using milvus::test::MilvusServerTest;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

class MilvusServerTestWithAuth : public MilvusServerTest {
 protected:
    std::shared_ptr<milvus::MilvusClient> root_client_;

    void
    SetUp() override {
        server_.SetAuthorizationEnabled(true);
        MilvusServerTest::SetUp();
        root_client_ = milvus::MilvusClient::Create();
        root_client_->Connect({"127.0.0.1", server_.ListenPort(), "root", "Milvus"});
    }

    void
    TearDown() override {
        MilvusServerTest::TearDown();
    }
};

TEST_F(MilvusServerTestWithAuth, GenericTest) {
    bool has;
    auto status = root_client_->HasCollection("nosuchcollection", has);
    EXPECT_TRUE(status.IsOk());
    // orig client with out user/pass
    status = client_->HasCollection("nosuchcollection", has);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::NOT_CONNECTED);
}
