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

using ::milvus::StatusCode;
using ::milvus::proto::milvus::SelectRoleRequest;
using ::milvus::proto::milvus::SelectRoleResponse;
using ::testing::_;
using ::testing::ElementsAreArray;

TEST_F(MilvusMockedTest, ListRoles) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::vector<std::string> expected_names{"role_1", "role_2"};

    EXPECT_CALL(service_, SelectRole(_, _, _))
        .WillOnce(
            [&expected_names](::grpc::ServerContext*, const SelectRoleRequest* request, SelectRoleResponse* response) {
                EXPECT_TRUE(request->role().name().empty());

                for (auto& name : expected_names) {
                    auto res = response->add_results();
                    res->mutable_role()->set_name(name);
                }
                return ::grpc::Status{};
            });

    std::vector<std::string> names;
    auto status = client_->ListRoles(names);
    EXPECT_TRUE(status.IsOk());

    EXPECT_THAT(names, ElementsAreArray(expected_names));
}
