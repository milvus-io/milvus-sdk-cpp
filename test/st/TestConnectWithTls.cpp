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

#include <array>
#include <cstdlib>
#include <random>

#include "MilvusServerTest.h"
#include "milvus/MilvusClient.h"
#include "milvus/Status.h"
#include "milvus/types/ConnectParam.h"

using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

static void
generate_certificates() {
    std::system("mkdir -p certs");
    std::system("openssl genrsa -out certs/ca.key 2048");
    std::system(
        "openssl req -new"
        " -key certs/ca.key"
        " -subj /C=CN/ST=Zhejiang/L=Hangzhou/O=Milvus/OU=CppSdk/CN=ca.test.com"
        " -out certs/ca.csr");
    std::system(
        "openssl x509 -req"
        " -days 365"
        " -in certs/ca.csr"
        " -signkey certs/ca.key"
        " -out certs/ca.crt");
    for (const auto& name : {"server", "client"}) {
        std::system((std::string("openssl genrsa -out certs/") + name + ".key 2048").c_str());
        std::system((std::string("openssl req -new -key certs/") + name +
                     ".key"
                     " -subj /C=CN/ST=Zhejiang/L=Hangzhou/O=Milvus/OU=CppSdk/CN=" +
                     name +
                     ".test.com"
                     " -out certs/" +
                     name + ".csr")
                        .c_str());
        std::system((std::string("openssl x509 -req -days 365 -in certs/") + name +
                     ".csr"
                     " -CA certs/ca.crt -CAkey certs/ca.key -CAcreateserial"
                     " -out certs/" +
                     name + ".crt")
                        .c_str());
    }
}

using milvus::test::MilvusServerTest;

template <int Mode>
class MilvusServerTestWithTlsMode : public MilvusServerTest {
 protected:
    std::shared_ptr<milvus::MilvusClient> ssl_client_;

    void
    SetUp() override {
        generate_certificates();
        std::array<char, 256> path;
        getcwd(path.data(), path.size());
        std::string pwd = path.data();
        server_.SetTls(Mode, pwd + "/certs/server.crt", pwd + "/certs/server.key", pwd + "/certs/ca.crt");

        MilvusServerTest::SetUp();

        ssl_client_ = milvus::MilvusClient::Create();
        auto param = server_.TestClientParam();
        ssl_client_->Connect(*param);
    }

    void
    TearDown() override {
        MilvusServerTest::TearDown();
        std::system("rm -fr certs/");
    }
};

class MilvusServerTestWithTlsMode1 : public MilvusServerTestWithTlsMode<1> {};
// TODO: fix it with milvus2.3+tls2
class DISABLED_MilvusServerTestWithTlsMode2 : public MilvusServerTestWithTlsMode<2> {};

TEST_F(MilvusServerTestWithTlsMode1, GenericTest) {
    bool has;
    auto status = ssl_client_->HasCollection("nosuchcollection", has);
    EXPECT_TRUE(status.IsOk());
    status = client_->HasCollection("nosuchcollection", has);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::NOT_CONNECTED);
}

TEST_F(DISABLED_MilvusServerTestWithTlsMode2, GenericTest) {
    bool has;
    auto status = ssl_client_->HasCollection("nosuchcollection", has);
    EXPECT_TRUE(status.IsOk());
    status = client_->HasCollection("nosuchcollection", has);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::NOT_CONNECTED);

    milvus::ConnectParam param{"127.0.0.1", 300};
    param.EnableTls();
    client_->Connect(param);
    status = client_->HasCollection("nosuchcollection", has);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::NOT_CONNECTED);
}
