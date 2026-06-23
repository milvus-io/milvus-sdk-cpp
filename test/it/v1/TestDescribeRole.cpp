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
using ::milvus::proto::milvus::SelectGrantRequest;
using ::milvus::proto::milvus::SelectGrantResponse;
using ::milvus::proto::milvus::SelectRoleRequest;
using ::milvus::proto::milvus::SelectRoleResponse;
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

TEST_F(UnconnectMilvusMockedTest, DescribeRole) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::RoleDesc expected_desc;
    expected_desc.SetName("Foo");
    expected_desc.AddGrantItem({"a", "b", "c", "d", "e", "f"});
    expected_desc.AddGrantItem({"1", "2", "3", "4", "5", "6"});

    EXPECT_CALL(service_, SelectGrant(_, _, _))
        .WillOnce(
            [&expected_desc](::grpc::ServerContext*, const SelectGrantRequest* request, SelectGrantResponse* response) {
                EXPECT_EQ(request->entity().role().name(), expected_desc.Name());

                for (const auto& item : expected_desc.GrantItems()) {
                    auto entity = response->mutable_entities()->Add();
                    entity->mutable_object()->set_name(item.object_type_);
                    entity->set_object_name(item.object_name_);
                    entity->set_db_name(item.db_name_);
                    entity->mutable_role()->set_name(item.role_name_);
                    entity->mutable_grantor()->mutable_user()->set_name(item.grantor_name_);
                    entity->mutable_grantor()->mutable_privilege()->set_name(item.privilege_);
                }
                return ::grpc::Status{};
            });

    EXPECT_CALL(service_, SelectRole(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SelectRoleRequest* request, SelectRoleResponse* response) {
            EXPECT_EQ(request->role().name(), "Foo");
            auto result = response->mutable_results()->Add();
            result->mutable_role()->set_name("Foo");
            result->mutable_role()->set_description("role description");
            return ::grpc::Status{};
        });

    milvus::DescribeRoleRequest request;
    request.WithRoleName(expected_desc.Name());
    milvus::DescribeRoleResponse response;
    auto status = client->DescribeRole(request, response);
    EXPECT_TRUE(status.IsOk());

    const auto& desc = response.Desc();
    EXPECT_EQ(desc.Name(), expected_desc.Name());
    EXPECT_EQ(desc.Description(), "role description");
    EXPECT_EQ(desc.GrantItems().size(), expected_desc.GrantItems().size());
    for (auto i = 0; i < desc.GrantItems().size(); i++) {
        EXPECT_EQ(desc.GrantItems().at(i).object_type_, expected_desc.GrantItems().at(i).object_type_);
        EXPECT_EQ(desc.GrantItems().at(i).object_name_, expected_desc.GrantItems().at(i).object_name_);
        EXPECT_EQ(desc.GrantItems().at(i).db_name_, expected_desc.GrantItems().at(i).db_name_);
        EXPECT_EQ(desc.GrantItems().at(i).role_name_, expected_desc.GrantItems().at(i).role_name_);
        EXPECT_EQ(desc.GrantItems().at(i).grantor_name_, expected_desc.GrantItems().at(i).grantor_name_);
        EXPECT_EQ(desc.GrantItems().at(i).privilege_, expected_desc.GrantItems().at(i).privilege_);
    }
}
