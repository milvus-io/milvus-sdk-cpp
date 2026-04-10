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

class FunctionTest : public ::testing::Test {};

TEST_F(FunctionTest, DefaultConstructor) {
    milvus::Function func;
    EXPECT_TRUE(func.Name().empty());
    EXPECT_EQ(func.GetFunctionType(), milvus::FunctionType::UNKNOWN);
    EXPECT_TRUE(func.InputFieldNames().empty());
    EXPECT_TRUE(func.OutputFieldNames().empty());
    EXPECT_TRUE(func.Params().empty());
}

TEST_F(FunctionTest, ParameterizedConstructor) {
    milvus::Function func("my_func", milvus::FunctionType::BM25, "a description");
    EXPECT_EQ(func.Name(), "my_func");
    EXPECT_EQ(func.GetFunctionType(), milvus::FunctionType::BM25);
    EXPECT_EQ(func.Description(), "a description");
}

TEST_F(FunctionTest, SetName) {
    milvus::Function func;
    auto status = func.SetName("func_name");
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(func.Name(), "func_name");
}

TEST_F(FunctionTest, SetDescription) {
    milvus::Function func;
    auto status = func.SetDescription("desc");
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(func.Description(), "desc");
}

TEST_F(FunctionTest, SetFunctionType) {
    milvus::Function func;
    auto status = func.SetFunctionType(milvus::FunctionType::RERANK);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(func.GetFunctionType(), milvus::FunctionType::RERANK);
}

TEST_F(FunctionTest, AddInputFieldName) {
    milvus::Function func;
    auto status = func.AddInputFieldName("input_1");
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(func.InputFieldNames().size(), 1);
    EXPECT_EQ(func.InputFieldNames()[0], "input_1");

    func.AddInputFieldName("input_2");
    EXPECT_EQ(func.InputFieldNames().size(), 2);
}

TEST_F(FunctionTest, AddOutputFieldName) {
    milvus::Function func;
    auto status = func.AddOutputFieldName("output_1");
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(func.OutputFieldNames().size(), 1);
    EXPECT_EQ(func.OutputFieldNames()[0], "output_1");
}

TEST_F(FunctionTest, AddParam) {
    milvus::Function func;
    auto status = func.AddParam("key1", "value1");
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(func.Params().size(), 1);
    EXPECT_EQ(func.Params().at("key1"), "value1");
}

class FunctionScoreTest : public ::testing::Test {};

TEST_F(FunctionScoreTest, DefaultConstructor) {
    milvus::FunctionScore fs;
    EXPECT_TRUE(fs.Functions().empty());
    EXPECT_TRUE(fs.Params().empty());
}

TEST_F(FunctionScoreTest, AddFunction) {
    milvus::FunctionScore fs;
    auto func = std::make_shared<milvus::Function>("f1", milvus::FunctionType::BM25);
    fs.AddFunction(func);
    EXPECT_EQ(fs.Functions().size(), 1);
    EXPECT_EQ(fs.Functions()[0]->Name(), "f1");
}

TEST_F(FunctionScoreTest, WithFunctions) {
    milvus::FunctionScore fs;
    std::vector<milvus::FunctionPtr> funcs;
    funcs.push_back(std::make_shared<milvus::Function>("f1", milvus::FunctionType::BM25));
    funcs.push_back(std::make_shared<milvus::Function>("f2", milvus::FunctionType::RERANK));
    fs.WithFunctions(std::move(funcs));
    EXPECT_EQ(fs.Functions().size(), 2);
}

TEST_F(FunctionScoreTest, SetFunctions) {
    milvus::FunctionScore fs;
    std::vector<milvus::FunctionPtr> funcs;
    funcs.push_back(std::make_shared<milvus::Function>("f1", milvus::FunctionType::BM25));
    fs.SetFunctions(std::move(funcs));
    EXPECT_EQ(fs.Functions().size(), 1);
}

TEST_F(FunctionScoreTest, AddParam) {
    milvus::FunctionScore fs;
    nlohmann::json val = 42;
    fs.AddParam("k", std::move(val));
    EXPECT_EQ(fs.Params().size(), 1);
    EXPECT_EQ(fs.Params().at("k"), 42);
}

TEST_F(FunctionScoreTest, WithParams) {
    milvus::FunctionScore fs;
    std::unordered_map<std::string, nlohmann::json> params;
    params["a"] = "b";
    fs.WithParams(std::move(params));
    EXPECT_EQ(fs.Params().size(), 1);
    EXPECT_EQ(fs.Params().at("a"), "b");
}

TEST_F(FunctionScoreTest, SetParams) {
    milvus::FunctionScore fs;
    std::unordered_map<std::string, nlohmann::json> params;
    params["x"] = 100;
    fs.SetParams(std::move(params));
    EXPECT_EQ(fs.Params().size(), 1);
    EXPECT_EQ(fs.Params().at("x"), 100);
}
