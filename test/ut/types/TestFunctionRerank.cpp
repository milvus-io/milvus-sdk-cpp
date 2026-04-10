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

class RRFRerankTest : public ::testing::Test {};

TEST_F(RRFRerankTest, DefaultConstructor) {
    milvus::RRFRerank rrf;
    EXPECT_EQ(rrf.GetFunctionType(), milvus::FunctionType::RERANK);
}

TEST_F(RRFRerankTest, ConstructorWithK) {
    milvus::RRFRerank rrf(60);
    EXPECT_EQ(rrf.GetFunctionType(), milvus::FunctionType::RERANK);
}

TEST_F(RRFRerankTest, SetK) {
    milvus::RRFRerank rrf;
    auto status = rrf.SetK(100);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(RRFRerankTest, SetFunctionTypeRejectNonRerank) {
    milvus::RRFRerank rrf;
    auto status = rrf.SetFunctionType(milvus::FunctionType::BM25);
    EXPECT_FALSE(status.IsOk());
}

TEST_F(RRFRerankTest, SetFunctionTypeAcceptRerank) {
    milvus::RRFRerank rrf;
    auto status = rrf.SetFunctionType(milvus::FunctionType::RERANK);
    EXPECT_TRUE(status.IsOk());
}

class WeightedRerankTest : public ::testing::Test {};

TEST_F(WeightedRerankTest, Constructor) {
    std::vector<float> weights = {0.5f, 0.3f, 0.2f};
    milvus::WeightedRerank wr(weights);
    EXPECT_EQ(wr.GetFunctionType(), milvus::FunctionType::RERANK);
}

TEST_F(WeightedRerankTest, SetWeights) {
    std::vector<float> weights = {1.0f};
    milvus::WeightedRerank wr(weights);
    std::vector<float> new_weights = {0.6f, 0.4f};
    auto status = wr.SetWeights(new_weights);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(WeightedRerankTest, SetFunctionTypeRejectNonRerank) {
    std::vector<float> weights = {1.0f};
    milvus::WeightedRerank wr(weights);
    auto status = wr.SetFunctionType(milvus::FunctionType::BM25);
    EXPECT_FALSE(status.IsOk());
}

class BoostRerankTest : public ::testing::Test {};

TEST_F(BoostRerankTest, Constructor) {
    milvus::BoostRerank br("boost_func");
    EXPECT_EQ(br.GetFunctionType(), milvus::FunctionType::RERANK);
    EXPECT_EQ(br.Name(), "boost_func");
}

TEST_F(BoostRerankTest, SetFilter) {
    milvus::BoostRerank br("boost_func");
    br.SetFilter("age > 18");
    // SetFilter modifies params, verify no crash
}

TEST_F(BoostRerankTest, SetWeight) {
    milvus::BoostRerank br("boost_func");
    br.SetWeight(2.5f);
    // SetWeight modifies params, verify no crash
}

TEST_F(BoostRerankTest, SetRandomScoreField) {
    milvus::BoostRerank br("boost_func");
    br.SetRandomScoreField("random_field");
    // verify no crash
}

TEST_F(BoostRerankTest, SetRandomScoreSeed) {
    milvus::BoostRerank br("boost_func");
    br.SetRandomScoreSeed(42);
    // verify no crash
}

TEST_F(BoostRerankTest, SetFunctionTypeRejectNonRerank) {
    milvus::BoostRerank br("boost_func");
    auto status = br.SetFunctionType(milvus::FunctionType::BM25);
    EXPECT_FALSE(status.IsOk());
}

class DecayRerankTest : public ::testing::Test {};

TEST_F(DecayRerankTest, Constructor) {
    milvus::DecayRerank dr("decay_func");
    EXPECT_EQ(dr.GetFunctionType(), milvus::FunctionType::RERANK);
    EXPECT_EQ(dr.Name(), "decay_func");
}

TEST_F(DecayRerankTest, SetFunction) {
    milvus::DecayRerank dr("decay_func");
    dr.SetFunction("gauss");
    // verify no crash
}

TEST_F(DecayRerankTest, SetOrigin) {
    milvus::DecayRerank dr("decay_func");
    dr.SetOrigin(100.0);
    EXPECT_EQ(dr.Params().at("origin"), std::to_string(100.0));
}

TEST_F(DecayRerankTest, SetOffset) {
    milvus::DecayRerank dr("decay_func");
    dr.SetOffset(10);
    EXPECT_EQ(dr.Params().at("offset"), std::to_string(10));
}

TEST_F(DecayRerankTest, SetScale) {
    milvus::DecayRerank dr("decay_func");
    dr.SetScale(50.0f);
    EXPECT_EQ(dr.Params().at("scale"), std::to_string(50.0f));
}

TEST_F(DecayRerankTest, SetDecay) {
    milvus::DecayRerank dr("decay_func");
    dr.SetDecay(0.5f);
    // verify no crash
}

TEST_F(DecayRerankTest, SetFunctionTypeRejectNonRerank) {
    milvus::DecayRerank dr("decay_func");
    auto status = dr.SetFunctionType(milvus::FunctionType::BM25);
    EXPECT_FALSE(status.IsOk());
}
