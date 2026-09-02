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

#include <memory>

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"

using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::GetVersionRequest;
using ::milvus::proto::milvus::GetVersionResponse;
using ::testing::_;

namespace {

std::shared_ptr<milvus::MilvusClientV2>
CreateConnectedV2Client(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{"127.0.0.1", port};
    auto status = client->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());
    return client;
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, GetServerVersionV2WithoutDetail) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetVersion(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetVersionRequest*, GetVersionResponse* response) {
            response->set_version("v2.5.0");
            return ::grpc::Status{};
        });

    milvus::GetServerVersionResponse response;
    auto status = client->GetServerVersionV2(milvus::GetServerVersionRequest(), response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.Version(), "v2.5.0");
    EXPECT_TRUE(response.BuildTime().empty());
}

TEST_F(UnconnectMilvusMockedTest, GetServerVersionV2WithDetail) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    // The second Connect RPC is issued for the detailed version query; it should carry
    // the SDK client_info for parity with the normal connect handshake.
    EXPECT_CALL(service_, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest* request, ConnectResponse* response) {
            EXPECT_EQ(request->client_info().sdk_type(), "CPP");
            EXPECT_FALSE(request->client_info().sdk_version().empty());
            auto* info = response->mutable_server_info();
            info->set_build_tags("v2.5.0");
            info->set_build_time("2024-01-01 00:00:00");
            info->set_git_commit("abc123");
            info->set_go_version("go1.21");
            info->set_deploy_mode("standalone");
            return ::grpc::Status{};
        });

    milvus::GetServerVersionResponse response;
    auto status = client->GetServerVersionV2(milvus::GetServerVersionRequest().WithDetail(true), response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.Version(), "v2.5.0");
    EXPECT_EQ(response.BuildTime(), "2024-01-01 00:00:00");
    EXPECT_EQ(response.GitCommit(), "abc123");
    EXPECT_EQ(response.GoVersion(), "go1.21");
    EXPECT_EQ(response.DeployMode(), "standalone");
}
