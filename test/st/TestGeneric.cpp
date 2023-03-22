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

#include "MilvusServerTest.h"
#include "gmock/gmock.h"

using milvus::test::MilvusServerTest;

class MilvusServerTestGeneric : public MilvusServerTest {};

TEST_F(MilvusServerTestGeneric, GetVersion) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string version;
    auto status = client_->GetVersion(version);
    EXPECT_TRUE(status.IsOk());
    EXPECT_THAT(version, testing::StartsWith("v2."));
}
