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

#include "../mocks/MilvusMockedTest.h"

using ::milvus::proto::milvus::DropRoleRequest;
using ::testing::_;

TEST_F(MilvusMockedTest, DropRole) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string role_name = "Foo";
    bool force_drop = true;

    EXPECT_CALL(service_, DropRole(_, _, _))
        .WillOnce([&role_name, &force_drop](::grpc::ServerContext*, const DropRoleRequest* request,
                                            ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->role_name(), role_name);
            EXPECT_EQ(request->force_drop(), force_drop);
            return ::grpc::Status{};
        });

    auto status = client_->DropRole(role_name, force_drop);
    EXPECT_TRUE(status.IsOk());
}
