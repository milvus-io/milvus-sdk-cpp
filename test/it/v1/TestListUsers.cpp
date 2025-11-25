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
using ::milvus::proto::milvus::ListCredUsersRequest;
using ::milvus::proto::milvus::ListCredUsersResponse;
using ::testing::_;
using ::testing::ElementsAre;

TEST_F(MilvusMockedTest, ListUsers) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    EXPECT_CALL(service_, ListCredUsers(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListCredUsersRequest* request, ListCredUsersResponse* resp) {
            resp->add_usernames("foo");
            resp->add_usernames("bar");
            return ::grpc::Status{};
        });
    std::vector<std::string> users;
    auto status = client_->ListUsers(users);

    EXPECT_TRUE(status.IsOk());
    EXPECT_THAT(users, ElementsAre("foo", "bar"));
}
