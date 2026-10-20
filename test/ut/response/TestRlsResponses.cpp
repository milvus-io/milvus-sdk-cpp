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

class ListRowPoliciesResponseTest : public ::testing::Test {};

TEST_F(ListRowPoliciesResponseTest, SetterAndGetter) {
    milvus::ListRowPoliciesResponse resp;

    milvus::RowPolicy policy;
    policy.SetPolicyName("policy_1");
    policy.SetPolicyType(milvus::RowPolicyType::PERMISSIVE);
    std::vector<milvus::RowPolicyAction> actions{milvus::RowPolicyAction::QUERY};
    policy.SetActions(std::move(actions));
    policy.SetUsingExpr("A > 1");
    policy.SetCheckExpr("");
    policy.SetDescription("desc");
    policy.SetCreatedAt(1234);

    std::vector<milvus::RowPolicy> policies;
    policies.push_back(policy);
    resp.SetPolicies(std::move(policies));

    ASSERT_EQ(resp.Policies().size(), 1);
    EXPECT_EQ(resp.Policies()[0].PolicyName(), "policy_1");
    EXPECT_EQ(resp.Policies()[0].PolicyType(), milvus::RowPolicyType::PERMISSIVE);
    ASSERT_EQ(resp.Policies()[0].Actions().size(), 1);
    EXPECT_EQ(resp.Policies()[0].Actions()[0], milvus::RowPolicyAction::QUERY);
    EXPECT_EQ(resp.Policies()[0].UsingExpr(), "A > 1");
    EXPECT_TRUE(resp.Policies()[0].CheckExpr().empty());
    EXPECT_EQ(resp.Policies()[0].Description(), "desc");
    EXPECT_EQ(resp.Policies()[0].CreatedAt(), 1234);
}

class GetRLSPrincipalTagsResponseTest : public ::testing::Test {};

TEST_F(GetRLSPrincipalTagsResponseTest, SetterAndGetter) {
    milvus::GetRLSPrincipalTagsResponse resp;
    nlohmann::json tags = {{"department", "finance"}, {"level", 3}, {"group", {"a", "b"}}};
    resp.SetTags(std::move(tags));
    resp.SetPrincipalName("alice");
    EXPECT_EQ(resp.Tags().at("department"), "finance");
    EXPECT_EQ(resp.Tags().at("level"), 3);
    EXPECT_EQ(resp.Tags().at("group"), nlohmann::json::array({"a", "b"}));
    EXPECT_EQ(resp.PrincipalName(), "alice");
}

class ListRLSPrincipalsResponseTest : public ::testing::Test {};

TEST_F(ListRLSPrincipalsResponseTest, SetterAndGetter) {
    milvus::ListRLSPrincipalsResponse resp;
    std::vector<std::string> principals{"alice", "bob"};
    resp.SetPrincipalNames(std::move(principals));
    ASSERT_EQ(resp.PrincipalNames().size(), 2);
    EXPECT_EQ(resp.PrincipalNames()[0], "alice");
    EXPECT_EQ(resp.PrincipalNames()[1], "bob");
}
