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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::AlterRoleRequest;
using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::CreateCredentialRequest;
using ::milvus::proto::milvus::CreateRoleRequest;
using ::milvus::proto::milvus::CreateRowPolicyRequest;
using ::milvus::proto::milvus::DeleteRLSPrincipalTagsRequest;
using ::milvus::proto::milvus::DropRowPolicyRequest;
using ::milvus::proto::milvus::GetRLSPrincipalTagsRequest;
using ::milvus::proto::milvus::GetRLSPrincipalTagsResponse;
using ::milvus::proto::milvus::ListRLSPrincipalsRequest;
using ::milvus::proto::milvus::ListRLSPrincipalsResponse;
using ::milvus::proto::milvus::ListRowPoliciesRequest;
using ::milvus::proto::milvus::ListRowPoliciesResponse;
using ::milvus::proto::milvus::OperatePrivilegeRequest;
using ::milvus::proto::milvus::SelectGrantRequest;
using ::milvus::proto::milvus::SelectGrantResponse;
using ::milvus::proto::milvus::SelectRoleRequest;
using ::milvus::proto::milvus::SelectRoleResponse;
using ::milvus::proto::milvus::SelectUserRequest;
using ::milvus::proto::milvus::SelectUserResponse;
using ::milvus::proto::milvus::SetRLSPrincipalTagsRequest;
using ::milvus::proto::milvus::UpdateCredentialRequest;
using ::milvus::proto::milvus::UpdateRowPolicyRequest;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Property;

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
TEST_F(UnconnectMilvusMockedTest, CreateRowPolicy) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, CreateRowPolicy(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const CreateRowPolicyRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->db_name(), "my_db");
            EXPECT_EQ(request->collection_name(), "my_coll");
            EXPECT_EQ(request->policy_name(), "policy_1");
            EXPECT_EQ(request->actions_size(), 2);
            EXPECT_EQ(request->actions(0), ::milvus::proto::milvus::RowPolicyAction::Query);
            EXPECT_EQ(request->actions(1), ::milvus::proto::milvus::RowPolicyAction::Search);
            EXPECT_EQ(request->using_expr(), "A > 1");
            EXPECT_EQ(request->check_expr(), "B < 2");
            EXPECT_EQ(request->description(), "policy description");
            EXPECT_EQ(request->policy_type(), ::milvus::proto::milvus::RowPolicyType::RowPolicyTypePermissive);
            EXPECT_TRUE(request->has_policy_type());
            return ::grpc::Status{};
        });

    std::vector<milvus::RowPolicyAction> actions{milvus::RowPolicyAction::QUERY, milvus::RowPolicyAction::SEARCH};
    auto status = client->CreateRowPolicy(milvus::CreateRowPolicyRequest()
                                              .WithDatabaseName("my_db")
                                              .WithCollectionName("my_coll")
                                              .WithPolicyName("policy_1")
                                              .WithPolicyType(milvus::RowPolicyType::PERMISSIVE)
                                              .WithActions(std::move(actions))
                                              .WithUsingExpr("A > 1")
                                              .WithCheckExpr("B < 2")
                                              .WithDescription("policy description"));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, CreateRowPolicyUnsetType) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    // The client rejects an unset policy type before any RPC is issued.
    std::vector<milvus::RowPolicyAction> actions{milvus::RowPolicyAction::QUERY};
    auto status = client->CreateRowPolicy(milvus::CreateRowPolicyRequest()
                                              .WithCollectionName("my_coll")
                                              .WithPolicyName("policy_1")
                                              .WithActions(std::move(actions)));
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, UpdateRowPolicy) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, UpdateRowPolicy(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const UpdateRowPolicyRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->db_name(), "my_db");
            EXPECT_EQ(request->collection_name(), "my_coll");
            EXPECT_EQ(request->policy_name(), "policy_1");
            EXPECT_EQ(request->actions_size(), 1);
            EXPECT_EQ(request->actions(0), ::milvus::proto::milvus::RowPolicyAction::Delete);
            EXPECT_EQ(request->using_expr(), "U");
            EXPECT_EQ(request->check_expr(), "C");
            EXPECT_EQ(request->description(), "D");
            EXPECT_EQ(request->policy_type(), ::milvus::proto::milvus::RowPolicyType::RowPolicyTypeRestrictive);
            return ::grpc::Status{};
        });

    std::vector<milvus::RowPolicyAction> actions{milvus::RowPolicyAction::DELETE_ROWS};
    auto status = client->UpdateRowPolicy(milvus::UpdateRowPolicyRequest()
                                              .WithDatabaseName("my_db")
                                              .WithCollectionName("my_coll")
                                              .WithPolicyName("policy_1")
                                              .WithPolicyType(milvus::RowPolicyType::RESTRICTIVE)
                                              .WithActions(std::move(actions))
                                              .WithUsingExpr("U")
                                              .WithCheckExpr("C")
                                              .WithDescription("D"));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, DropRowPolicy) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DropRowPolicy(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const DropRowPolicyRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->db_name(), "my_db");
            EXPECT_EQ(request->collection_name(), "my_coll");
            EXPECT_EQ(request->policy_name(), "policy_1");
            return ::grpc::Status{};
        });

    auto status = client->DropRowPolicy(
        milvus::DropRowPolicyRequest().WithDatabaseName("my_db").WithCollectionName("my_coll").WithPolicyName(
            "policy_1"));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, ListRowPolicies) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListRowPolicies(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListRowPoliciesRequest* request, ListRowPoliciesResponse* response) {
            EXPECT_EQ(request->db_name(), "my_db");
            EXPECT_EQ(request->collection_name(), "my_coll");

            auto policy = response->add_policies();
            policy->set_policy_name("policy_1");
            policy->add_actions(::milvus::proto::milvus::RowPolicyAction::Query);
            policy->set_using_expr("A > 1");
            policy->set_description("desc");
            policy->set_created_at(1234);
            policy->set_policy_type(::milvus::proto::milvus::RowPolicyType::RowPolicyTypePermissive);
            return ::grpc::Status{};
        });

    milvus::ListRowPoliciesRequest request;
    request.WithDatabaseName("my_db").WithCollectionName("my_coll");
    milvus::ListRowPoliciesResponse response;
    auto status = client->ListRowPolicies(request, response);
    EXPECT_TRUE(status.IsOk());

    ASSERT_EQ(response.Policies().size(), 1);
    EXPECT_EQ(response.Policies()[0].PolicyName(), "policy_1");
    EXPECT_EQ(response.Policies()[0].PolicyType(), milvus::RowPolicyType::PERMISSIVE);
    ASSERT_EQ(response.Policies()[0].Actions().size(), 1);
    EXPECT_EQ(response.Policies()[0].Actions()[0], milvus::RowPolicyAction::QUERY);
    EXPECT_EQ(response.Policies()[0].UsingExpr(), "A > 1");
    EXPECT_TRUE(response.Policies()[0].CheckExpr().empty());
    EXPECT_EQ(response.Policies()[0].Description(), "desc");
    EXPECT_EQ(response.Policies()[0].CreatedAt(), 1234);
}

TEST_F(UnconnectMilvusMockedTest, SetRLSPrincipalTags) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, SetRLSPrincipalTags(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const SetRLSPrincipalTagsRequest* request, ::milvus::proto::common::Status*) {
                EXPECT_EQ(request->db_name(), "my_db");
                EXPECT_EQ(request->collection_name(), "my_coll");
                EXPECT_EQ(request->principal_name(), "alice");
                auto tags = nlohmann::json::parse(request->tags());
                EXPECT_EQ(tags.at("department"), "finance");
                EXPECT_EQ(tags.at("level"), 3);
                EXPECT_EQ(tags.at("group"), nlohmann::json::array({"a", "b"}));
                return ::grpc::Status{};
            });

    nlohmann::json tags = {{"department", "finance"}, {"level", 3}, {"group", {"a", "b"}}};
    auto status = client->SetRLSPrincipalTags(milvus::SetRLSPrincipalTagsRequest()
                                                  .WithDatabaseName("my_db")
                                                  .WithCollectionName("my_coll")
                                                  .WithPrincipalName("alice")
                                                  .WithTags(tags));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, SetRLSPrincipalTagsInvalidTags) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    std::string invalid_utf8;
    invalid_utf8.push_back(static_cast<char>(0xff));
    invalid_utf8.push_back(static_cast<char>(0xfe));

    const std::vector<nlohmann::json> invalid_tags = {
        nlohmann::json::array({"a", "b"}),                  // non-object top level
        nlohmann::json::object(),                           // empty object
        {{"k", true}},                                      // boolean value
        {{"k", nullptr}},                                   // null value
        {{"k", {{"nested", 1}}}},                           // object value
        {{"k", std::numeric_limits<double>::quiet_NaN()}},  // NaN number
        {{"k", std::numeric_limits<double>::infinity()}},   // Infinity number
        {{"k", {{"a", "b"}, {"c", "d"}}}},                  // nested (two-dimensional) array
        {{"k", nlohmann::json::array({1, {{"x", 2}}})}},    // array with an object element
        {{"k", invalid_utf8}},                              // invalid UTF-8 string, dump() throws
    };

    for (const auto& tags : invalid_tags) {
        auto status = client->SetRLSPrincipalTags(
            milvus::SetRLSPrincipalTagsRequest().WithCollectionName("my_coll").WithPrincipalName("alice").WithTags(
                tags));
        EXPECT_FALSE(status.IsOk());
        EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
    }
}

TEST_F(UnconnectMilvusMockedTest, GetRLSPrincipalTagsMalformedOrEmptyTags) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetRLSPrincipalTags(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetRLSPrincipalTagsRequest*, GetRLSPrincipalTagsResponse* response) {
            response->set_tags("not json {");
            return ::grpc::Status{};
        })
        .WillOnce([](::grpc::ServerContext*, const GetRLSPrincipalTagsRequest*, GetRLSPrincipalTagsResponse* response) {
            response->set_tags("");
            return ::grpc::Status{};
        });

    milvus::GetRLSPrincipalTagsRequest request;
    request.WithCollectionName("my_coll").WithPrincipalName("alice");
    milvus::GetRLSPrincipalTagsResponse response;
    auto status = client->GetRLSPrincipalTags(request, response);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::JSON_PARSE_ERROR);

    status = client->GetRLSPrincipalTags(request, response);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::JSON_PARSE_ERROR);
}

TEST_F(UnconnectMilvusMockedTest, GetRLSPrincipalTags) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetRLSPrincipalTags(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetRLSPrincipalTagsRequest* request,
                     GetRLSPrincipalTagsResponse* response) {
            EXPECT_EQ(request->db_name(), "my_db");
            EXPECT_EQ(request->collection_name(), "my_coll");
            EXPECT_EQ(request->principal_name(), "alice");
            response->set_tags(R"({"department":"finance","level":3,"group":["a","b"]})");
            return ::grpc::Status{};
        });

    milvus::GetRLSPrincipalTagsRequest request;
    request.WithDatabaseName("my_db").WithCollectionName("my_coll").WithPrincipalName("alice");
    milvus::GetRLSPrincipalTagsResponse response;
    auto status = client->GetRLSPrincipalTags(request, response);
    EXPECT_TRUE(status.IsOk());

    EXPECT_EQ(response.PrincipalName(), "alice");
    EXPECT_EQ(response.Tags().at("department"), "finance");
    EXPECT_EQ(response.Tags().at("level"), 3);
    EXPECT_EQ(response.Tags().at("group"), nlohmann::json::array({"a", "b"}));
}

TEST_F(UnconnectMilvusMockedTest, ListRLSPrincipals) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListRLSPrincipals(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const ListRLSPrincipalsRequest* request, ListRLSPrincipalsResponse* response) {
                EXPECT_EQ(request->db_name(), "my_db");
                EXPECT_EQ(request->collection_name(), "my_coll");
                response->add_principal_names("alice");
                response->add_principal_names("bob");
                return ::grpc::Status{};
            });

    milvus::ListRLSPrincipalsRequest request;
    request.WithDatabaseName("my_db").WithCollectionName("my_coll");
    milvus::ListRLSPrincipalsResponse response;
    auto status = client->ListRLSPrincipals(request, response);
    EXPECT_TRUE(status.IsOk());

    ASSERT_EQ(response.PrincipalNames().size(), 2);
    EXPECT_EQ(response.PrincipalNames()[0], "alice");
    EXPECT_EQ(response.PrincipalNames()[1], "bob");
}

TEST_F(UnconnectMilvusMockedTest, DeleteRLSPrincipalTags) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DeleteRLSPrincipalTags(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const DeleteRLSPrincipalTagsRequest* request, ::milvus::proto::common::Status*) {
                EXPECT_EQ(request->db_name(), "my_db");
                EXPECT_EQ(request->collection_name(), "my_coll");
                EXPECT_EQ(request->principal_name(), "alice");
                EXPECT_EQ(request->tag_keys_size(), 2);
                EXPECT_EQ(request->tag_keys(0), "tag_1");
                EXPECT_EQ(request->tag_keys(1), "tag_2");
                return ::grpc::Status{};
            });

    std::vector<std::string> tag_keys{"tag_1", "tag_2"};
    auto status = client->DeleteRLSPrincipalTags(milvus::DeleteRLSPrincipalTagsRequest()
                                                     .WithDatabaseName("my_db")
                                                     .WithCollectionName("my_coll")
                                                     .WithPrincipalName("alice")
                                                     .WithTagKeys(std::move(tag_keys)));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, DeleteRLSPrincipalTagsAllTags) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DeleteRLSPrincipalTags(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const DeleteRLSPrincipalTagsRequest* request, ::milvus::proto::common::Status*) {
                EXPECT_EQ(request->db_name(), "my_db");
                EXPECT_EQ(request->collection_name(), "my_coll");
                EXPECT_EQ(request->principal_name(), "alice");
                EXPECT_EQ(request->tag_keys_size(), 0);
                return ::grpc::Status{};
            });

    auto status = client->DeleteRLSPrincipalTags(milvus::DeleteRLSPrincipalTagsRequest()
                                                     .WithDatabaseName("my_db")
                                                     .WithCollectionName("my_coll")
                                                     .WithPrincipalName("alice"));
    EXPECT_TRUE(status.IsOk());
}
