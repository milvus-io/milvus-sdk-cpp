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
using ::milvus::proto::milvus::SelectUserRequest;
using ::milvus::proto::milvus::SelectUserResponse;
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

TEST_F(UnconnectMilvusMockedTest, DescribeUser) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::UserDesc expected_desc;
    expected_desc.SetName("Bar");
    expected_desc.AddRole("role_1");
    expected_desc.AddRole("role_2");

    EXPECT_CALL(service_, SelectUser(_, _, _))
        .WillOnce(
            [&expected_desc](::grpc::ServerContext*, const SelectUserRequest* request, SelectUserResponse* response) {
                EXPECT_EQ(request->user().name(), expected_desc.Name());
                EXPECT_TRUE(request->include_role_info());

                auto result = response->mutable_results()->Add();
                result->mutable_user()->set_name(expected_desc.Name());
                result->set_description("user description");
                for (const auto& role : expected_desc.Roles()) {
                    result->add_roles()->set_name(role);
                }
                return ::grpc::Status{};
            });

    milvus::DescribeUserRequest request;
    request.WithUserName(expected_desc.Name());
    milvus::DescribeUserResponse response;
    auto status = client->DescribeUser(request, response);
    EXPECT_TRUE(status.IsOk());

    const auto& desc = response.Desc();
    EXPECT_EQ(desc.Name(), expected_desc.Name());
    EXPECT_EQ(desc.Description(), "user description");
    EXPECT_EQ(desc.Roles().size(), expected_desc.Roles().size());
    for (auto i = 0; i < desc.Roles().size(); i++) {
        EXPECT_EQ(desc.Roles().at(i), expected_desc.Roles().at(i));
    }
}
