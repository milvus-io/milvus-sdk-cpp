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

#include "milvus/types/ConnectParam.h"

class ConnectParamTest : public ::testing::Test {};

TEST_F(ConnectParamTest, GeneralTesting) {
    milvus::ConnectParam param{"localhost", 10000};
    EXPECT_EQ(param.Host(), "localhost");
    EXPECT_EQ(param.Port(), 10000);
    EXPECT_EQ(param.Uri(), "localhost:10000");
    EXPECT_EQ(param.ConnectTimeout(), 10000);

    param.SetConnectTimeout(1000);
    EXPECT_EQ(param.ConnectTimeout(), 1000);

    param.EnableTls();
    EXPECT_TRUE(param.TlsEnabled());

    param.EnableTls("local", "ca");
    EXPECT_TRUE(param.TlsEnabled());
    EXPECT_EQ(param.ServerName(), "local");
    EXPECT_EQ(param.CaCert(), "ca");
    EXPECT_EQ(param.Cert(), "");
    EXPECT_EQ(param.Key(), "");

    param.EnableTls("local", "a", "b", "c");
    EXPECT_EQ(param.Cert(), "a");
    EXPECT_EQ(param.Key(), "b");

    param.DisableTls();
    EXPECT_FALSE(param.TlsEnabled());
}

TEST_F(ConnectParamTest, KeepaliveDefaults) {
    milvus::ConnectParam param{"localhost", 19530};

    EXPECT_EQ(param.KeepaliveTimeMs(), 10000);
    EXPECT_EQ(param.KeepaliveTimeoutMs(), 5000);
    EXPECT_TRUE(param.KeepaliveWithoutCalls());
}

TEST_F(ConnectParamTest, KeepaliveSettersAndBuilders) {
    milvus::ConnectParam param{"localhost", 19530};

    param.SetKeepaliveTimeMs(20000);
    EXPECT_EQ(param.KeepaliveTimeMs(), 20000);

    param.SetKeepaliveTimeoutMs(8000);
    EXPECT_EQ(param.KeepaliveTimeoutMs(), 8000);

    param.SetKeepaliveWithoutCalls(false);
    EXPECT_FALSE(param.KeepaliveWithoutCalls());

    // Test builder pattern
    auto& ref = param.WithKeepaliveTimeMs(30000).WithKeepaliveTimeoutMs(10000).WithKeepaliveWithoutCalls(true);
    EXPECT_EQ(ref.KeepaliveTimeMs(), 30000);
    EXPECT_EQ(ref.KeepaliveTimeoutMs(), 10000);
    EXPECT_TRUE(ref.KeepaliveWithoutCalls());
}
