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
#include <unistd.h>

#include <cstdlib>
#include <random>

#include "MilvusServerTest.h"

using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

using milvus::test::MilvusServerTest;

class MilvusServerTestWithTlsMode : public MilvusServerTest {};

TEST_F(MilvusServerTestWithTlsMode, GenericTest) {
    bool has;
    auto status = client_->HasCollection("nosuchcollection", has);
    EXPECT_TRUE(status.IsOk());
    EXPECT_FALSE(has);

    // client without certifications
    milvus::ConnectParam param{"127.0.0.1", 19530};
    param.EnableTls();
    std::shared_ptr<milvus::MilvusClient> tempClient = milvus::MilvusClient::Create();
    status = tempClient->Connect(param);
    EXPECT_FALSE(status.IsOk());
    status = tempClient->HasCollection("nosuchcollection", has);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::NOT_CONNECTED);
}
