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

class FunctionChainTest : public ::testing::Test {};

TEST_F(FunctionChainTest, ColumnRefAndHelper) {
    auto ref = milvus::col("$score");
    EXPECT_EQ(ref.Name(), "$score");
}

TEST_F(FunctionChainTest, ExprArgColumnAndLiteral) {
    milvus::FunctionChainExprArg column_arg(milvus::col("$score"));
    EXPECT_TRUE(column_arg.IsColumn());
    EXPECT_FALSE(column_arg.IsLiteral());
    EXPECT_EQ(column_arg.ColumnName(), "$score");

    milvus::FunctionChainExprArg literal_arg(nlohmann::json(3.14));
    EXPECT_TRUE(literal_arg.IsLiteral());
    EXPECT_FALSE(literal_arg.IsColumn());
    EXPECT_EQ(literal_arg.Literal().get<double>(), 3.14);
}

TEST_F(FunctionChainTest, ExprBuild) {
    milvus::FunctionChainExpr expr("round_decimal");
    expr.AddColumnArg("$score").AddParam("decimal", 4);
    EXPECT_EQ(expr.Name(), "round_decimal");
    ASSERT_EQ(expr.Args().size(), 1u);
    EXPECT_TRUE(expr.Args()[0].IsColumn());
    EXPECT_EQ(expr.Params().at("decimal").get<int>(), 4);
}

TEST_F(FunctionChainTest, MapSortLimitChain) {
    milvus::FunctionChainExpr expr("round_decimal");
    expr.AddColumnArg("$score").AddParam("decimal", 4);

    milvus::FunctionChain chain(milvus::FunctionChainStage::L2_RERANK, "rerank");
    chain.Map("$score", expr).Sort("$score", true, "$id").Limit(10, 0);

    EXPECT_EQ(chain.Stage(), milvus::FunctionChainStage::L2_RERANK);
    EXPECT_EQ(chain.Name(), "rerank");
    ASSERT_EQ(chain.Ops().size(), 3u);

    EXPECT_EQ(chain.Ops()[0].Op(), "map");
    EXPECT_TRUE(chain.Ops()[0].HasExpr());
    EXPECT_EQ(chain.Ops()[0].Outputs().size(), 1u);
    EXPECT_EQ(chain.Ops()[0].Outputs()[0], "$score");

    EXPECT_EQ(chain.Ops()[1].Op(), "sort");
    EXPECT_EQ(chain.Ops()[1].Params().at("desc").get<bool>(), true);

    EXPECT_EQ(chain.Ops()[2].Op(), "limit");
    EXPECT_EQ(chain.Ops()[2].Params().at("limit").get<int64_t>(), 10);
}

TEST_F(FunctionChainTest, SearchRequestFunctionChains) {
    milvus::SearchRequest request;
    milvus::FunctionChain chain(milvus::FunctionChainStage::L2_RERANK);
    chain.Limit(5);
    request.AddFunctionChain(chain);

    ASSERT_EQ(request.FunctionChains().size(), 1u);
    EXPECT_EQ(request.FunctionChains()[0].Ops().size(), 1u);
}

TEST_F(FunctionChainTest, FunctionChainsAndRerankConflict) {
    milvus::SearchRequest request;
    request.AddFloatVector(std::vector<float>(1, 1.0f));
    request.SetRerank(std::make_shared<milvus::FunctionScore>());
    milvus::FunctionChain chain(milvus::FunctionChainStage::L2_RERANK);
    request.AddFunctionChain(chain);

    auto status = request.Validate();
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
}

TEST_F(FunctionChainTest, UnspecifiedStageRejected) {
    milvus::SearchRequest request;
    request.AddFloatVector(std::vector<float>(1, 1.0f));
    milvus::FunctionChain chain(milvus::FunctionChainStage::UNSPECIFIED);
    chain.Limit(5);
    request.AddFunctionChain(chain);

    auto status = request.Validate();
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
}

TEST_F(FunctionChainTest, EmptyOpNameRejected) {
    milvus::SearchRequest request;
    request.AddFloatVector(std::vector<float>(1, 1.0f));
    milvus::FunctionChain chain(milvus::FunctionChainStage::L2_RERANK);
    chain.AddOp(milvus::FunctionChainOp(""));
    request.AddFunctionChain(chain);

    EXPECT_FALSE(request.Validate().IsOk());
}

TEST_F(FunctionChainTest, EmptyExprNameRejected) {
    milvus::SearchRequest request;
    request.AddFloatVector(std::vector<float>(1, 1.0f));
    milvus::FunctionChain chain(milvus::FunctionChainStage::L2_RERANK);
    chain.Map("$score", milvus::FunctionChainExpr(""));
    request.AddFunctionChain(chain);

    EXPECT_FALSE(request.Validate().IsOk());
}

TEST_F(FunctionChainTest, EmptyColumnNameRejected) {
    milvus::SearchRequest request;
    request.AddFloatVector(std::vector<float>(1, 1.0f));
    milvus::FunctionChainExpr expr("round_decimal");
    expr.AddColumnArg("");
    milvus::FunctionChain chain(milvus::FunctionChainStage::L2_RERANK);
    chain.Map("$score", expr);
    request.AddFunctionChain(chain);

    EXPECT_FALSE(request.Validate().IsOk());
}
