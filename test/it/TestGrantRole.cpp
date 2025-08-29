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

using ::milvus::proto::milvus::OperateUserRoleRequest;
using ::testing::_;

TEST_F(MilvusMockedTest, GrantRole) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string user_name = "Foo";
    const std::string role_name = "Bar";

    EXPECT_CALL(service_, OperateUserRole(_, _, _))
        .WillOnce([&user_name, &role_name](::grpc::ServerContext*, const OperateUserRoleRequest* request,
                                           ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->username(), user_name);
            EXPECT_EQ(request->role_name(), role_name);
            EXPECT_EQ(request->type(), ::milvus::proto::milvus::OperateUserRoleType::AddUserToRole);

            return ::grpc::Status{};
        });

    auto status = client_->GrantRole(user_name, role_name);
    EXPECT_TRUE(status.IsOk());
}
