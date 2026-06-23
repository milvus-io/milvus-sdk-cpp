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

#include <nlohmann/json.hpp>

#include "milvus/types/Highlighter.h"

class LexicalHighlighterTest : public ::testing::Test {};

TEST_F(LexicalHighlighterTest, DefaultState) {
    milvus::LexicalHighlighter highlighter;
    EXPECT_EQ(highlighter.HighlightType(), "Lexical");
    EXPECT_TRUE(highlighter.Params().empty());
}

TEST_F(LexicalHighlighterTest, SerializeParams) {
    milvus::LexicalHighlighter highlighter;

    auto& ref = highlighter.WithHighlightQueries({{"term", "title", "milvus"}, {"phrase", "body", "vector db"}})
                    .WithHighlightSearchText(true)
                    .WithPreTags({"<em>", "<strong>"})
                    .WithPostTags({"</em>", "</strong>"})
                    .WithFragmentOffset(1)
                    .WithFragmentSize(16)
                    .WithNumOfFragments(2);

    EXPECT_EQ(&ref, &highlighter);

    const auto& params = highlighter.Params();
    EXPECT_EQ(params.size(), 7UL);
    EXPECT_EQ(params.at("highlight_search_text"), "true");
    EXPECT_EQ(params.at("fragment_offset"), "1");
    EXPECT_EQ(params.at("fragment_size"), "16");
    EXPECT_EQ(params.at("num_of_fragments"), "2");

    auto queries = nlohmann::json::parse(params.at("highlight_query"));
    ASSERT_TRUE(queries.is_array());
    ASSERT_EQ(queries.size(), 2UL);
    EXPECT_EQ(queries[0]["type"], "term");
    EXPECT_EQ(queries[0]["field"], "title");
    EXPECT_EQ(queries[0]["text"], "milvus");
    EXPECT_EQ(queries[1]["type"], "phrase");
    EXPECT_EQ(queries[1]["field"], "body");
    EXPECT_EQ(queries[1]["text"], "vector db");

    auto pre_tags = nlohmann::json::parse(params.at("pre_tags"));
    ASSERT_TRUE(pre_tags.is_array());
    ASSERT_EQ(pre_tags.size(), 2UL);
    EXPECT_EQ(pre_tags[0], "<em>");
    EXPECT_EQ(pre_tags[1], "<strong>");

    auto post_tags = nlohmann::json::parse(params.at("post_tags"));
    ASSERT_TRUE(post_tags.is_array());
    ASSERT_EQ(post_tags.size(), 2UL);
    EXPECT_EQ(post_tags[0], "</em>");
    EXPECT_EQ(post_tags[1], "</strong>");
}

class SemanticHighlighterTest : public ::testing::Test {};

TEST_F(SemanticHighlighterTest, DefaultState) {
    milvus::SemanticHighlighter highlighter;
    EXPECT_EQ(highlighter.HighlightType(), "Semantic");
    EXPECT_TRUE(highlighter.Params().empty());
}

TEST_F(SemanticHighlighterTest, SerializeParams) {
    milvus::SemanticHighlighter highlighter;

    auto& ref = highlighter.WithQueries({"milvus sdk", "vector database"})
                    .WithInputFields({"title", "body"})
                    .WithPreTags({"<mark>"})
                    .WithPostTags({"</mark>"})
                    .WithThreshold(0.8f)
                    .WithHighlightOnly(false)
                    .WithModelDeploymentID("deploy-1")
                    .WithMaxClientBatchSize(64);

    EXPECT_EQ(&ref, &highlighter);

    const auto& params = highlighter.Params();
    EXPECT_EQ(params.size(), 8UL);
    EXPECT_EQ(params.at("threshold"), std::to_string(0.8f));
    EXPECT_EQ(params.at("highlight_only"), "false");
    EXPECT_EQ(params.at("model_deployment_id"), "deploy-1");
    EXPECT_EQ(params.at("max_client_batch_size"), "64");

    auto queries = nlohmann::json::parse(params.at("queries"));
    ASSERT_TRUE(queries.is_array());
    ASSERT_EQ(queries.size(), 2UL);
    EXPECT_EQ(queries[0], "milvus sdk");
    EXPECT_EQ(queries[1], "vector database");

    auto input_fields = nlohmann::json::parse(params.at("input_fields"));
    ASSERT_TRUE(input_fields.is_array());
    ASSERT_EQ(input_fields.size(), 2UL);
    EXPECT_EQ(input_fields[0], "title");
    EXPECT_EQ(input_fields[1], "body");

    auto pre_tags = nlohmann::json::parse(params.at("pre_tags"));
    ASSERT_TRUE(pre_tags.is_array());
    ASSERT_EQ(pre_tags.size(), 1UL);
    EXPECT_EQ(pre_tags[0], "<mark>");

    auto post_tags = nlohmann::json::parse(params.at("post_tags"));
    ASSERT_TRUE(post_tags.is_array());
    ASSERT_EQ(post_tags.size(), 1UL);
    EXPECT_EQ(post_tags[0], "</mark>");
}
