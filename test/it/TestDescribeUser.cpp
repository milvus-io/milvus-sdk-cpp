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

#include "mocks/MilvusMockedTest.h"

using ::milvus::proto::milvus::SelectUserRequest;
using ::milvus::proto::milvus::SelectUserResponse;
using ::testing::_;

TEST_F(MilvusMockedTest, DescribeUser) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

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
                for (const auto& role : expected_desc.Roles()) {
                    result->add_roles()->set_name(role);
                }
                return ::grpc::Status{};
            });

    milvus::UserDesc desc;
    auto status = client_->DescribeUser(expected_desc.Name(), desc);
    EXPECT_TRUE(status.IsOk());

    EXPECT_EQ(desc.Name(), expected_desc.Name());
    EXPECT_EQ(desc.Roles().size(), expected_desc.Roles().size());
    for (auto i = 0; i < desc.Roles().size(); i++) {
        EXPECT_EQ(desc.Roles().at(i), expected_desc.Roles().at(i));
    }
}
