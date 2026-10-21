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

#include "milvus/MilvusClientV2.h"

class CreateRowPolicyRequestTest : public ::testing::Test {};

TEST_F(CreateRowPolicyRequestTest, GettersAndSetters) {
    milvus::CreateRowPolicyRequest req;

    req.WithDatabaseName("my_db");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    req.WithCollectionName("my_coll");
    EXPECT_EQ(req.CollectionName(), "my_coll");

    req.WithPolicyName("policy_1");
    EXPECT_EQ(req.PolicyName(), "policy_1");

    req.WithPolicyType(milvus::RowPolicyType::PERMISSIVE);
    EXPECT_EQ(req.PolicyType(), milvus::RowPolicyType::PERMISSIVE);

    std::vector<milvus::RowPolicyAction> actions{milvus::RowPolicyAction::QUERY, milvus::RowPolicyAction::SEARCH};
    req.WithActions(std::move(actions));
    EXPECT_EQ(req.Actions().size(), 2);
    EXPECT_EQ(req.Actions()[0], milvus::RowPolicyAction::QUERY);
    EXPECT_EQ(req.Actions()[1], milvus::RowPolicyAction::SEARCH);

    req.WithUsingExpr("$current_principal == \"alice\"");
    EXPECT_EQ(req.UsingExpr(), "$current_principal == \"alice\"");

    req.WithCheckExpr("");
    EXPECT_TRUE(req.CheckExpr().empty());

    req.WithDescription("policy description");
    EXPECT_EQ(req.Description(), "policy description");

    req.SetDatabaseName("default");
    req.SetCollectionName("coll_2");
    req.SetPolicyName("policy_2");
    req.SetPolicyType(milvus::RowPolicyType::RESTRICTIVE);
    req.SetUsingExpr("A > 1");
    req.SetCheckExpr("B < 2");
    req.SetDescription("updated description");
    EXPECT_EQ(req.DatabaseName(), "default");
    EXPECT_EQ(req.CollectionName(), "coll_2");
    EXPECT_EQ(req.PolicyName(), "policy_2");
    EXPECT_EQ(req.PolicyType(), milvus::RowPolicyType::RESTRICTIVE);
    EXPECT_EQ(req.UsingExpr(), "A > 1");
    EXPECT_EQ(req.CheckExpr(), "B < 2");
    EXPECT_EQ(req.Description(), "updated description");
}

TEST_F(CreateRowPolicyRequestTest, FluentChaining) {
    milvus::CreateRowPolicyRequest req;
    auto& ref = req.WithDatabaseName("d")
                    .WithCollectionName("c")
                    .WithPolicyName("p")
                    .WithPolicyType(milvus::RowPolicyType::PERMISSIVE)
                    .WithUsingExpr("U")
                    .WithCheckExpr("C")
                    .WithDescription("desc");
    EXPECT_EQ(&ref, &req);
    EXPECT_EQ(req.DatabaseName(), "d");
    EXPECT_EQ(req.CollectionName(), "c");
    EXPECT_EQ(req.PolicyName(), "p");
    EXPECT_EQ(req.PolicyType(), milvus::RowPolicyType::PERMISSIVE);
    EXPECT_EQ(req.UsingExpr(), "U");
    EXPECT_EQ(req.CheckExpr(), "C");
    EXPECT_EQ(req.Description(), "desc");
}

TEST_F(CreateRowPolicyRequestTest, SetActions) {
    milvus::CreateRowPolicyRequest req;
    std::vector<milvus::RowPolicyAction> actions{milvus::RowPolicyAction::INSERT, milvus::RowPolicyAction::UPSERT};
    req.SetActions(std::move(actions));
    EXPECT_EQ(req.Actions().size(), 2);
    EXPECT_EQ(req.Actions()[0], milvus::RowPolicyAction::INSERT);
    EXPECT_EQ(req.Actions()[1], milvus::RowPolicyAction::UPSERT);
}

class UpdateRowPolicyRequestTest : public ::testing::Test {};

TEST_F(UpdateRowPolicyRequestTest, GettersAndSetters) {
    milvus::UpdateRowPolicyRequest req;

    req.WithDatabaseName("my_db");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    req.WithCollectionName("my_coll");
    EXPECT_EQ(req.CollectionName(), "my_coll");

    req.WithPolicyName("policy_1");
    EXPECT_EQ(req.PolicyName(), "policy_1");

    req.WithPolicyType(milvus::RowPolicyType::RESTRICTIVE);
    EXPECT_EQ(req.PolicyType(), milvus::RowPolicyType::RESTRICTIVE);

    std::vector<milvus::RowPolicyAction> actions{milvus::RowPolicyAction::DELETE_ROWS};
    req.SetActions(std::move(actions));
    EXPECT_EQ(req.Actions().size(), 1);
    EXPECT_EQ(req.Actions()[0], milvus::RowPolicyAction::DELETE_ROWS);

    req.WithUsingExpr("U").WithCheckExpr("C").WithDescription("D");
    EXPECT_EQ(req.UsingExpr(), "U");
    EXPECT_EQ(req.CheckExpr(), "C");
    EXPECT_EQ(req.Description(), "D");
}

class DropRowPolicyRequestTest : public ::testing::Test {};

TEST_F(DropRowPolicyRequestTest, GettersAndSetters) {
    milvus::DropRowPolicyRequest req;
    req.WithDatabaseName("my_db");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    req.WithCollectionName("my_coll");
    EXPECT_EQ(req.CollectionName(), "my_coll");

    req.WithPolicyName("policy_1");
    EXPECT_EQ(req.PolicyName(), "policy_1");

    req.SetDatabaseName("default");
    req.SetCollectionName("coll_2");
    req.SetPolicyName("policy_2");
    EXPECT_EQ(req.DatabaseName(), "default");
    EXPECT_EQ(req.CollectionName(), "coll_2");
    EXPECT_EQ(req.PolicyName(), "policy_2");
}

class ListRowPoliciesRequestTest : public ::testing::Test {};

TEST_F(ListRowPoliciesRequestTest, GettersAndSetters) {
    milvus::ListRowPoliciesRequest req;
    req.WithDatabaseName("my_db");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    req.WithCollectionName("my_coll");
    EXPECT_EQ(req.CollectionName(), "my_coll");

    req.SetDatabaseName("default");
    req.SetCollectionName("coll_2");
    EXPECT_EQ(req.DatabaseName(), "default");
    EXPECT_EQ(req.CollectionName(), "coll_2");
}

class SetRLSPrincipalTagsRequestTest : public ::testing::Test {};

TEST_F(SetRLSPrincipalTagsRequestTest, GettersAndSetters) {
    milvus::SetRLSPrincipalTagsRequest req;

    req.WithDatabaseName("my_db");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    req.WithCollectionName("my_coll");
    EXPECT_EQ(req.CollectionName(), "my_coll");

    req.WithPrincipalName("alice");
    EXPECT_EQ(req.PrincipalName(), "alice");

    nlohmann::json tags = {{"department", "finance"}, {"level", 3}};
    req.WithTags(tags);
    EXPECT_EQ(req.Tags(), tags);

    req.SetDatabaseName("default");
    req.SetCollectionName("coll_2");
    req.SetPrincipalName("bob");
    nlohmann::json tags2 = {{"group", {"a", "b"}}};
    req.SetTags(tags2);
    EXPECT_EQ(req.DatabaseName(), "default");
    EXPECT_EQ(req.CollectionName(), "coll_2");
    EXPECT_EQ(req.PrincipalName(), "bob");
    EXPECT_EQ(req.Tags(), tags2);
}

class GetRLSPrincipalTagsRequestTest : public ::testing::Test {};

TEST_F(GetRLSPrincipalTagsRequestTest, GettersAndSetters) {
    milvus::GetRLSPrincipalTagsRequest req;
    req.WithDatabaseName("my_db");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    req.WithCollectionName("my_coll");
    EXPECT_EQ(req.CollectionName(), "my_coll");

    req.WithPrincipalName("alice");
    EXPECT_EQ(req.PrincipalName(), "alice");

    req.SetPrincipalName("bob");
    EXPECT_EQ(req.PrincipalName(), "bob");
}

class ListRLSPrincipalsRequestTest : public ::testing::Test {};

TEST_F(ListRLSPrincipalsRequestTest, GettersAndSetters) {
    milvus::ListRLSPrincipalsRequest req;
    req.WithDatabaseName("my_db");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    req.WithCollectionName("my_coll");
    EXPECT_EQ(req.CollectionName(), "my_coll");

    req.SetDatabaseName("default");
    req.SetCollectionName("coll_2");
    EXPECT_EQ(req.DatabaseName(), "default");
    EXPECT_EQ(req.CollectionName(), "coll_2");
}

class DeleteRLSPrincipalTagsRequestTest : public ::testing::Test {};

TEST_F(DeleteRLSPrincipalTagsRequestTest, GettersAndSetters) {
    milvus::DeleteRLSPrincipalTagsRequest req;

    req.WithDatabaseName("my_db");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    req.WithCollectionName("my_coll");
    EXPECT_EQ(req.CollectionName(), "my_coll");

    req.WithPrincipalName("alice");
    EXPECT_EQ(req.PrincipalName(), "alice");

    std::vector<std::string> tag_keys{"tag_1", "tag_2"};
    req.WithTagKeys(std::move(tag_keys));
    EXPECT_EQ(req.TagKeys().size(), 2);
    EXPECT_EQ(req.TagKeys()[0], "tag_1");
    EXPECT_EQ(req.TagKeys()[1], "tag_2");

    req.SetPrincipalName("bob");
    std::vector<std::string> tag_keys2;
    req.SetTagKeys(std::move(tag_keys2));
    EXPECT_EQ(req.PrincipalName(), "bob");
    EXPECT_TRUE(req.TagKeys().empty());
}
