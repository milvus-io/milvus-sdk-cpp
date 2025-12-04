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
#include "milvus/types/Constants.h"

using ::milvus::proto::milvus::OperatePrivilegeGroupRequest;
using ::milvus::proto::milvus::OperatePrivilegeGroupType;
using ::testing::_;

TEST_F(MilvusMockedTest, RemovePrivilegesFromGroup) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string group_name = "Foo";
    const std::vector<std::string> privileges{"a", "bb", "ccc"};

    EXPECT_CALL(service_, OperatePrivilegeGroup(_, _, _))
        .WillOnce([&group_name, &privileges](::grpc::ServerContext*, const OperatePrivilegeGroupRequest* request,
                                             ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->group_name(), group_name);
            EXPECT_EQ(request->type(), OperatePrivilegeGroupType::RemovePrivilegesFromGroup);
            EXPECT_EQ(request->privileges_size(), privileges.size());
            for (auto i = 0; i < privileges.size(); i++) {
                EXPECT_EQ(request->privileges().at(i).name(), privileges.at(i));
            }
            return ::grpc::Status{};
        });

    auto status = client_->RemovePrivilegesFromGroup(group_name, privileges);
    EXPECT_TRUE(status.IsOk());
}
