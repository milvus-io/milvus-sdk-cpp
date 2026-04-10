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

TEST_F(ConnectParamTest, UriSetterAndBuilder) {
    milvus::ConnectParam param{"localhost", 19530};
    param.SetUri("http://remote:19530");
    EXPECT_EQ(param.Uri(), "http://remote:19530");

    auto& ref = param.WithUri("http://another:19530");
    EXPECT_EQ(ref.Uri(), "http://another:19530");
}

TEST_F(ConnectParamTest, UriConstructor) {
    milvus::ConnectParam param("http://myhost:19530");
    EXPECT_EQ(param.Uri(), "http://myhost:19530");
}

TEST_F(ConnectParamTest, TokenSetterAndBuilder) {
    milvus::ConnectParam param{"localhost", 19530};
    EXPECT_EQ(param.Token(), "");

    param.SetToken("my_token");
    EXPECT_EQ(param.Token(), "my_token");

    auto& ref = param.WithToken("another_token");
    EXPECT_EQ(ref.Token(), "another_token");
}

TEST_F(ConnectParamTest, UriTokenConstructor) {
    milvus::ConnectParam param("http://myhost:19530", "my_token");
    EXPECT_EQ(param.Uri(), "http://myhost:19530");
    EXPECT_EQ(param.Token(), "my_token");
}

TEST_F(ConnectParamTest, AuthorizationsSetterAndBuilder) {
    milvus::ConnectParam param{"localhost", 19530};
    EXPECT_EQ(param.Authorizations(), "");

    param.SetAuthorizations("user", "pass");
    EXPECT_FALSE(param.Authorizations().empty());
    EXPECT_EQ(param.Username(), "user");

    auto& ref = param.WithAuthorizations("user2", "pass2");
    EXPECT_FALSE(ref.Authorizations().empty());
    EXPECT_EQ(ref.Username(), "user2");
}

TEST_F(ConnectParamTest, HostPortUsernamePasswordConstructor) {
    milvus::ConnectParam param("localhost", 19530, "admin", "password");
    EXPECT_EQ(param.Username(), "admin");
    EXPECT_FALSE(param.Authorizations().empty());
}

TEST_F(ConnectParamTest, DbNameSetterAndBuilder) {
    milvus::ConnectParam param{"localhost", 19530};
    EXPECT_EQ(param.DbName(), "");

    param.SetDbName("mydb");
    EXPECT_EQ(param.DbName(), "mydb");

    auto& ref = param.WithDbName("otherdb");
    EXPECT_EQ(ref.DbName(), "otherdb");
}

TEST_F(ConnectParamTest, RpcDeadlineMsSetterAndBuilder) {
    milvus::ConnectParam param{"localhost", 19530};
    EXPECT_EQ(param.RpcDeadlineMs(), 0);

    param.SetRpcDeadlineMs(5000);
    EXPECT_EQ(param.RpcDeadlineMs(), 5000);

    auto& ref = param.WithRpcDeadlineMs(10000);
    EXPECT_EQ(ref.RpcDeadlineMs(), 10000);
}

TEST_F(ConnectParamTest, WithTlsBuilder) {
    milvus::ConnectParam param{"localhost", 19530};
    EXPECT_FALSE(param.TlsEnabled());

    auto& ref = param.WithTls();
    EXPECT_TRUE(ref.TlsEnabled());

    param.DisableTls();
    EXPECT_FALSE(param.TlsEnabled());

    auto& ref2 = param.WithTls("server", "ca_cert");
    EXPECT_TRUE(ref2.TlsEnabled());
    EXPECT_EQ(ref2.ServerName(), "server");
    EXPECT_EQ(ref2.CaCert(), "ca_cert");

    param.DisableTls();
    auto& ref3 = param.WithTls("server", "cert", "key", "ca");
    EXPECT_TRUE(ref3.TlsEnabled());
    EXPECT_EQ(ref3.ServerName(), "server");
    EXPECT_EQ(ref3.Cert(), "cert");
    EXPECT_EQ(ref3.Key(), "key");
    EXPECT_EQ(ref3.CaCert(), "ca");
}
