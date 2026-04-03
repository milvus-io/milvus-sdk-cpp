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

#include "MilvusServerTest.h"
#include "gmock/gmock.h"

using milvus::test::MilvusServerTest;
class MilvusServerTestGeneric : public MilvusServerTest {};

TEST_F(MilvusServerTestGeneric, GetServerVersion) {
    std::string version;
    auto status = client_->GetServerVersion(version);
    std::cout << "Milvus version: " << version << std::endl;
    milvus::test::ExpectStatusOK(status);
    EXPECT_THAT(version, testing::MatchesRegex("v?2.+"));
}

TEST_F(MilvusServerTestGeneric, GetSDKVersion) {
    std::string version;
    auto status = client_->GetSDKVersion(version);
    std::cout << "SDK version: " << version << std::endl;
    milvus::test::ExpectStatusOK(status);
    EXPECT_FALSE(version.empty());
}

TEST_F(MilvusServerTestGeneric, CheckHealth) {
    milvus::CheckHealthResponse resp;
    auto status = client_->CheckHealth(milvus::CheckHealthRequest(), resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_TRUE(resp.IsHealthy());
}

TEST_F(MilvusServerTestGeneric, RunAnalyzer) {
    // helper lambda to run analyzer and verify results
    auto runAndVerify = [this](const nlohmann::json& params, const std::string& text, size_t min_tokens) {
        milvus::RunAnalyzerRequest req;
        req.AddText(text).WithAnalyzerParams(params).WithDetail(true).WithHash(true);

        milvus::RunAnalyzerResponse resp;
        auto status = client_->RunAnalyzer(req, resp);
        milvus::test::ExpectStatusOK(status);

        const auto& results = resp.Results();
        EXPECT_EQ(results.size(), 1);
        EXPECT_GE(results.at(0).Tokens().size(), min_tokens);
        return results.at(0).Tokens();
    };

    // standard tokenizer
    {
        nlohmann::json params = {{"type", "standard"}};
        auto tokens = runAndVerify(params, "hello world, milvus is great", 4);
    }

    // standard tokenizer with stop word filter
    {
        nlohmann::json params = {
            {"tokenizer", "standard"},
            {"filter", {{{"type", "stop"}, {"stop_words", {"and", "for"}}}}},
        };
        auto tokens = runAndVerify(params, "Milvus supports L2 distance and IP similarity for float vector.", 5);
        // "and" and "for" should be filtered out
        for (const auto& token : tokens) {
            EXPECT_NE(token.token_, "and");
            EXPECT_NE(token.token_, "for");
        }
    }

    // standard tokenizer with length filter
    {
        nlohmann::json params = {
            {"tokenizer", "standard"},
            {"filter", {{{"type", "length"}, {"max", 6}}}},
        };
        auto tokens = runAndVerify(params, "The length filter allows control over token length", 3);
        for (const auto& token : tokens) {
            EXPECT_LE(token.token_.size(), 6);
        }
    }

    // standard tokenizer with stemmer filter
    {
        nlohmann::json params = {
            {"tokenizer", "standard"},
            {"filter", {{{"type", "stemmer"}, {"language", "english"}}}},
        };
        auto tokens = runAndVerify(params, "running runs looked ran runner", 5);
    }

    // jieba tokenizer for Chinese
    {
        nlohmann::json params = {{"tokenizer", "jieba"}, {"filter", {"cnalphanumonly"}}};
        auto tokens = runAndVerify(params, "Milvus 是一个开源的向量数据库", 1);
    }

    // icu tokenizer for non-Latin scripts
    {
        nlohmann::json params = {{"tokenizer", "icu"}};
        auto tokens = runAndVerify(params, "Привет! Как дела?", 2);
    }

    // standard tokenizer with decompounder filter
    {
        nlohmann::json params = {
            {"tokenizer", "standard"},
            {"filter",
             {{{"type", "decompounder"}, {"word_list", {"dampf", "schiff", "fahrt", "brot", "backen", "automat"}}}}},
        };
        auto tokens = runAndVerify(params, "dampfschifffahrt brotbackautomat", 2);
    }

    // multiple texts in one request
    {
        milvus::RunAnalyzerRequest req;
        req.AddText("hello world");
        req.AddText("milvus vector database");
        req.AddText("running quickly");
        req.WithAnalyzerParams(nlohmann::json{{"type", "standard"}});

        milvus::RunAnalyzerResponse resp;
        auto status = client_->RunAnalyzer(req, resp);
        milvus::test::ExpectStatusOK(status);

        const auto& results = resp.Results();
        EXPECT_EQ(results.size(), 3);
        EXPECT_EQ(results.at(0).Tokens().size(), 2);  // "hello", "world"
        EXPECT_EQ(results.at(1).Tokens().size(), 3);  // "milvus", "vector", "database"
        EXPECT_EQ(results.at(2).Tokens().size(), 2);  // "running", "quickly"
    }
}
