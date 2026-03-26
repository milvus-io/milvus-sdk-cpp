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

TEST_F(MilvusServerTestGeneric, GetServerVersion) {
    std::string version;
    auto status = client_->GetServerVersion(version);
    std::cout << "Milvus version: " << version << std::endl;
    milvus::test::ExpectStatusOK(status);
    EXPECT_THAT(version, testing::MatchesRegex("v?2.+"));
}

TEST_F(MilvusServerTestGeneric, GetSDKVersion) {
    std::string version;
    auto status = client_->GetSDKVersion(version);
    std::cout << "SDK version: " << version << std::endl;
    milvus::test::ExpectStatusOK(status);
    EXPECT_FALSE(version.empty());
}

TEST_F(MilvusServerTestGeneric, CheckHealth) {
    milvus::CheckHealthResponse resp;
    auto status = client_->CheckHealth(milvus::CheckHealthRequest(), resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_TRUE(resp.IsHealthy());
}
