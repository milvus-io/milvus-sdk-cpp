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

class AnalyzerTokenTest : public ::testing::Test {};

TEST_F(AnalyzerTokenTest, StructFields) {
    milvus::AnalyzerToken token;
    token.token_ = "hello";
    token.start_offset_ = 0;
    token.end_offset_ = 5;
    token.position_ = 0;
    token.position_length_ = 1;
    token.hash_ = 12345;

    EXPECT_EQ(token.token_, "hello");
    EXPECT_EQ(token.start_offset_, 0);
    EXPECT_EQ(token.end_offset_, 5);
    EXPECT_EQ(token.position_, 0);
    EXPECT_EQ(token.position_length_, 1);
    EXPECT_EQ(token.hash_, 12345u);
}

class AnalyzerResultTest : public ::testing::Test {};

TEST_F(AnalyzerResultTest, ConstructorAndTokens) {
    std::vector<milvus::AnalyzerToken> tokens;
    milvus::AnalyzerToken t1;
    t1.token_ = "world";
    t1.start_offset_ = 6;
    t1.end_offset_ = 11;
    t1.position_ = 1;
    t1.position_length_ = 1;
    t1.hash_ = 99;
    tokens.push_back(t1);

    milvus::AnalyzerResult result(std::move(tokens));
    EXPECT_EQ(result.Tokens().size(), 1);
    EXPECT_EQ(result.Tokens()[0].token_, "world");
    EXPECT_EQ(result.Tokens()[0].hash_, 99u);
}

TEST_F(AnalyzerResultTest, AddToken) {
    std::vector<milvus::AnalyzerToken> empty_tokens;
    milvus::AnalyzerResult result(std::move(empty_tokens));
    EXPECT_EQ(result.Tokens().size(), 0);

    milvus::AnalyzerToken t;
    t.token_ = "test";
    t.start_offset_ = 0;
    t.end_offset_ = 4;
    t.position_ = 0;
    t.position_length_ = 1;
    t.hash_ = 42;

    auto status = result.AddToken(std::move(t));
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(result.Tokens().size(), 1);
    EXPECT_EQ(result.Tokens()[0].token_, "test");
}
