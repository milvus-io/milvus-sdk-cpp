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

class ModelRerankTest : public ::testing::Test {};

TEST_F(ModelRerankTest, Constructor) {
    milvus::ModelRerank mr("model_func");
    EXPECT_EQ(mr.GetFunctionType(), milvus::FunctionType::RERANK);
    EXPECT_EQ(mr.Name(), "model_func");
}

TEST_F(ModelRerankTest, SetFunctionTypeRejectNonRerank) {
    milvus::ModelRerank mr("model_func");
    auto status = mr.SetFunctionType(milvus::FunctionType::BM25);
    EXPECT_FALSE(status.IsOk());
}

TEST_F(ModelRerankTest, SetFunctionTypeAcceptRerank) {
    milvus::ModelRerank mr("model_func");
    auto status = mr.SetFunctionType(milvus::FunctionType::RERANK);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(ModelRerankTest, SetProvider) {
    milvus::ModelRerank mr("model_func");
    mr.SetProvider("cohere");
    EXPECT_EQ(mr.Params().at("provider"), "cohere");
}

TEST_F(ModelRerankTest, SetQueries) {
    milvus::ModelRerank mr("model_func");
    std::vector<std::string> queries{"what is AI?", "machine learning basics"};
    mr.SetQueries(queries);
    // Queries are stored in params; verify no crash and params updated
    EXPECT_FALSE(mr.Params().empty());
}

TEST_F(ModelRerankTest, SetEndpoint) {
    milvus::ModelRerank mr("model_func");
    mr.SetEndpoint("https://api.example.com/rerank");
    EXPECT_EQ(mr.Params().at("endpoint"), "https://api.example.com/rerank");
}

TEST_F(ModelRerankTest, SetMaxClientBatchSize) {
    milvus::ModelRerank mr("model_func");
    mr.SetMaxClientBatchSize(100);
    EXPECT_EQ(mr.Params().at("maxBatch"), std::to_string(100));
}
