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

using ::milvus::proto::milvus::ListPrivilegeGroupsRequest;
using ::milvus::proto::milvus::ListPrivilegeGroupsResponse;
using ::testing::_;
using ::testing::ElementsAreArray;

TEST_F(MilvusMockedTest, ListPrivilegeGroups) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    milvus::PrivilegeGroupInfos expected_groups;
    expected_groups.emplace_back("Foo", std::vector<std::string>{"a", "b"});
    expected_groups.emplace_back("Bar", std::vector<std::string>{"1"});

    EXPECT_CALL(service_, ListPrivilegeGroups(_, _, _))
        .WillOnce([&expected_groups](::grpc::ServerContext*, const ListPrivilegeGroupsRequest*,
                                     ListPrivilegeGroupsResponse* response) {
            for (auto& group : expected_groups) {
                auto rpc_group = response->mutable_privilege_groups()->Add();
                rpc_group->set_group_name(group.Name());
                for (auto& privilege : group.Privileges()) {
                    auto rpc_privilege = rpc_group->mutable_privileges()->Add();
                    rpc_privilege->set_name(privilege);
                }
            }
            return ::grpc::Status{};
        });

    milvus::PrivilegeGroupInfos groups;
    auto status = client_->ListPrivilegeGroups(groups);
    EXPECT_TRUE(status.IsOk());

    EXPECT_EQ(groups.size(), expected_groups.size());
    for (auto i = 0; i < groups.size(); i++) {
        EXPECT_EQ(groups.at(i).Name(), expected_groups.at(i).Name());
        EXPECT_THAT(groups.at(i).Privileges(), ElementsAreArray(expected_groups.at(i).Privileges()));
    }
}
