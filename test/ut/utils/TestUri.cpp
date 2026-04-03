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

#include "utils/Uri.h"

class UriTest : public ::testing::Test {};

TEST_F(UriTest, SimpleHostPort) {
    auto uri = milvus::ParseURI("localhost:19530");
    EXPECT_EQ(uri.scheme, "");
    EXPECT_EQ(uri.host, "localhost");
    EXPECT_EQ(uri.port, 19530);
    EXPECT_EQ(uri.path, "");
    EXPECT_EQ(uri.dbname, "");
}

TEST_F(UriTest, HttpScheme) {
    auto uri = milvus::ParseURI("http://localhost:19530");
    EXPECT_EQ(uri.scheme, "http");
    EXPECT_EQ(uri.host, "localhost");
    EXPECT_EQ(uri.port, 19530);
    EXPECT_EQ(uri.path, "");
    EXPECT_EQ(uri.dbname, "");
}

TEST_F(UriTest, HttpsScheme) {
    auto uri = milvus::ParseURI("https://myhost:443");
    EXPECT_EQ(uri.scheme, "https");
    EXPECT_EQ(uri.host, "myhost");
    EXPECT_EQ(uri.port, 443);
    EXPECT_EQ(uri.path, "");
    EXPECT_EQ(uri.dbname, "");
}

TEST_F(UriTest, WithPathDbname) {
    auto uri = milvus::ParseURI("http://localhost:19530/mydb");
    EXPECT_EQ(uri.scheme, "http");
    EXPECT_EQ(uri.host, "localhost");
    EXPECT_EQ(uri.port, 19530);
    EXPECT_EQ(uri.path, "/mydb");
    EXPECT_EQ(uri.dbname, "mydb");
}

TEST_F(UriTest, WithCredentials) {
    // The parser treats user:pass@localhost as the authority string.
    // The rfind(':') finds the colon before the port, so host = "user:pass@localhost" won't happen.
    // Actually with "user:pass@localhost:19530", rfind(':') is the port separator,
    // and find(':') is the first colon in user:pass, so multiple colons => treated as host without port.
    // Let's test what actually happens:
    auto uri = milvus::ParseURI("http://user:pass@localhost:19530");
    EXPECT_EQ(uri.scheme, "http");
    // Multiple colons in authority without brackets: the parser treats it as "no port"
    // because find(':') != rfind(':'), so the whole authority is the host.
    EXPECT_EQ(uri.host, "user:pass@localhost:19530");
    // Port defaults to 19530 for non-https when no port is explicitly parsed
    EXPECT_EQ(uri.port, 19530);
}

TEST_F(UriTest, IPv6) {
    auto uri = milvus::ParseURI("http://[::1]:19530");
    EXPECT_EQ(uri.scheme, "http");
    EXPECT_EQ(uri.host, "::1");
    EXPECT_EQ(uri.port, 19530);
    EXPECT_EQ(uri.path, "");
    EXPECT_EQ(uri.dbname, "");
}

TEST_F(UriTest, NoPort) {
    // No port with http scheme: defaults to 19530
    auto uri = milvus::ParseURI("http://localhost");
    EXPECT_EQ(uri.scheme, "http");
    EXPECT_EQ(uri.host, "localhost");
    EXPECT_EQ(uri.port, 19530);
    EXPECT_EQ(uri.path, "");
    EXPECT_EQ(uri.dbname, "");
}

TEST_F(UriTest, NoPortHttps) {
    // No port with https scheme: defaults to 443
    auto uri = milvus::ParseURI("https://localhost");
    EXPECT_EQ(uri.scheme, "https");
    EXPECT_EQ(uri.host, "localhost");
    EXPECT_EQ(uri.port, 443);
}

TEST_F(UriTest, JustHost) {
    // No scheme, no port: "myhost" is treated as authority with no colon, so default port
    auto uri = milvus::ParseURI("myhost");
    EXPECT_EQ(uri.scheme, "");
    EXPECT_EQ(uri.host, "myhost");
    EXPECT_EQ(uri.port, 19530);
    EXPECT_EQ(uri.path, "");
    EXPECT_EQ(uri.dbname, "");
}

TEST_F(UriTest, RootPath) {
    auto uri = milvus::ParseURI("http://localhost:19530/");
    EXPECT_EQ(uri.scheme, "http");
    EXPECT_EQ(uri.host, "localhost");
    EXPECT_EQ(uri.port, 19530);
    EXPECT_EQ(uri.path, "/");
    EXPECT_EQ(uri.dbname, "");
}

TEST_F(UriTest, IPv6NoBrackets) {
    // IPv6 without brackets: multiple colons, no single colon match => host = whole authority
    auto uri = milvus::ParseURI("::1");
    EXPECT_EQ(uri.scheme, "");
    EXPECT_EQ(uri.host, "::1");
    // no single colon found => defaults to 19530 (not https)
    EXPECT_EQ(uri.port, 19530);
}
