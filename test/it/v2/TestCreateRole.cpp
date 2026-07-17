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
using ::milvus::proto::milvus::CreateRoleRequest;
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

TEST_F(UnconnectMilvusMockedTest, CreateRole) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    const std::string role_name = "Foo";

    EXPECT_CALL(service_, CreateRole(_, _, _))
        .WillOnce(
            [&role_name](::grpc::ServerContext*, const CreateRoleRequest* request, ::milvus::proto::common::Status*) {
                EXPECT_EQ(request->entity().name(), role_name);
                EXPECT_EQ(request->entity().description(), "role description");
                return ::grpc::Status{};
            });

    milvus::CreateRoleRequest request;
    request.WithRoleName(role_name).WithDescription("role description");
    auto status = client->CreateRole(request);
    EXPECT_TRUE(status.IsOk());
}
