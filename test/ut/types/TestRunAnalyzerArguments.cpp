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

class RunAnalyzerArgumentsTest : public ::testing::Test {};

TEST_F(RunAnalyzerArgumentsTest, DefaultValues) {
    milvus::RunAnalyzerArguments args;
    EXPECT_TRUE(args.DatabaseName().empty());
    EXPECT_TRUE(args.CollectionName().empty());
    EXPECT_TRUE(args.FieldName().empty());
    EXPECT_TRUE(args.Texts().empty());
    EXPECT_TRUE(args.AnalyzerNames().empty());
    EXPECT_FALSE(args.IsWithDetail());
    EXPECT_FALSE(args.IsWithHash());
}

TEST_F(RunAnalyzerArgumentsTest, DatabaseName) {
    milvus::RunAnalyzerArguments args;
    auto status = args.SetDatabaseName("my_db");
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(args.DatabaseName(), "my_db");
}

TEST_F(RunAnalyzerArgumentsTest, WithDatabaseName) {
    milvus::RunAnalyzerArguments args;
    auto& ref = args.WithDatabaseName("db2");
    EXPECT_EQ(args.DatabaseName(), "db2");
    EXPECT_EQ(&ref, &args);
}

TEST_F(RunAnalyzerArgumentsTest, CollectionName) {
    milvus::RunAnalyzerArguments args;
    auto status = args.SetCollectionName("coll_1");
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(args.CollectionName(), "coll_1");
}

TEST_F(RunAnalyzerArgumentsTest, WithCollectionName) {
    milvus::RunAnalyzerArguments args;
    auto& ref = args.WithCollectionName("coll_2");
    EXPECT_EQ(args.CollectionName(), "coll_2");
    EXPECT_EQ(&ref, &args);
}

TEST_F(RunAnalyzerArgumentsTest, FieldName) {
    milvus::RunAnalyzerArguments args;
    auto status = args.SetFieldName("text_field");
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(args.FieldName(), "text_field");
}

TEST_F(RunAnalyzerArgumentsTest, WithFieldName) {
    milvus::RunAnalyzerArguments args;
    auto& ref = args.WithFieldName("field_2");
    EXPECT_EQ(args.FieldName(), "field_2");
    EXPECT_EQ(&ref, &args);
}

TEST_F(RunAnalyzerArgumentsTest, SetTexts) {
    milvus::RunAnalyzerArguments args;
    std::vector<std::string> texts = {"hello", "world"};
    auto status = args.SetTexts(texts);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(args.Texts().size(), 2);
    EXPECT_EQ(args.Texts()[0], "hello");
    EXPECT_EQ(args.Texts()[1], "world");
}

TEST_F(RunAnalyzerArgumentsTest, WithTexts) {
    milvus::RunAnalyzerArguments args;
    std::vector<std::string> texts = {"foo", "bar"};
    auto& ref = args.WithTexts(texts);
    EXPECT_EQ(args.Texts().size(), 2);
    EXPECT_EQ(&ref, &args);
}

TEST_F(RunAnalyzerArgumentsTest, AddText) {
    milvus::RunAnalyzerArguments args;
    auto& ref = args.AddText("text1");
    EXPECT_EQ(args.Texts().size(), 1);
    EXPECT_EQ(args.Texts()[0], "text1");
    EXPECT_EQ(&ref, &args);

    args.AddText("text2");
    EXPECT_EQ(args.Texts().size(), 2);
}

TEST_F(RunAnalyzerArgumentsTest, SetAnalyzerNames) {
    milvus::RunAnalyzerArguments args;
    std::vector<std::string> names = {"standard", "jieba"};
    auto status = args.SetAnalyzerNames(names);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(args.AnalyzerNames().size(), 2);
}

TEST_F(RunAnalyzerArgumentsTest, AddAnalyzerName) {
    milvus::RunAnalyzerArguments args;
    auto& ref = args.AddAnalyzerName("standard");
    EXPECT_EQ(args.AnalyzerNames().size(), 1);
    EXPECT_EQ(args.AnalyzerNames()[0], "standard");
    EXPECT_EQ(&ref, &args);
}

TEST_F(RunAnalyzerArgumentsTest, AnalyzerParams) {
    milvus::RunAnalyzerArguments args;
    nlohmann::json params = {{"type", "standard"}, {"max_token_length", 255}};
    auto status = args.SetAnalyzerParams(params);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(args.AnalyzerParams()["type"], "standard");
    EXPECT_EQ(args.AnalyzerParams()["max_token_length"], 255);
}

TEST_F(RunAnalyzerArgumentsTest, WithAnalyzerParams) {
    milvus::RunAnalyzerArguments args;
    nlohmann::json params = {{"key", "val"}};
    auto& ref = args.WithAnalyzerParams(params);
    EXPECT_EQ(args.AnalyzerParams()["key"], "val");
    EXPECT_EQ(&ref, &args);
}

TEST_F(RunAnalyzerArgumentsTest, WithDetail) {
    milvus::RunAnalyzerArguments args;
    auto& ref = args.WithDetail(true);
    EXPECT_TRUE(args.IsWithDetail());
    EXPECT_EQ(&ref, &args);

    args.WithDetail(false);
    EXPECT_FALSE(args.IsWithDetail());
}

TEST_F(RunAnalyzerArgumentsTest, WithHash) {
    milvus::RunAnalyzerArguments args;
    auto& ref = args.WithHash(true);
    EXPECT_TRUE(args.IsWithHash());
    EXPECT_EQ(&ref, &args);

    args.WithHash(false);
    EXPECT_FALSE(args.IsWithHash());
}
