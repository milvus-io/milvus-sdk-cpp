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

#include "mocks/MilvusMockedTest.h"

using ::milvus::proto::milvus::RunAnalyzerRequest;
using ::milvus::proto::milvus::RunAnalyzerResponse;
using ::testing::_;

TEST_F(MilvusMockedTest, RunAnalyzer) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string text = "dummy text";
    const std::string collection_name = "dummy coll";
    const std::string db_name = "dummy db";
    const std::string field_name = "dummy field";
    const std::string analyzer_name = "dummy analyzer";
    const nlohmann::json params = {
        {"tokenizer", "standard"},
        {"filter", {{{"type", "stop"}, {"stop_words", {"of"}}}}},
    };
    const bool with_detail = false;
    const bool with_hash = true;

    EXPECT_CALL(service_, RunAnalyzer(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const RunAnalyzerRequest* request, RunAnalyzerResponse* response) {
            EXPECT_EQ(request->collection_name(), collection_name);
            EXPECT_EQ(request->db_name(), db_name);
            EXPECT_EQ(request->field_name(), field_name);
            EXPECT_EQ(request->placeholder_size(), 1);
            EXPECT_EQ(request->placeholder().at(0), text);
            EXPECT_EQ(request->analyzer_names_size(), 1);
            EXPECT_EQ(request->analyzer_names().at(0), analyzer_name);
            EXPECT_EQ(request->analyzer_params(), params.dump());
            EXPECT_EQ(request->with_detail(), with_detail);
            EXPECT_EQ(request->with_hash(), with_hash);

            ::milvus::proto::milvus::AnalyzerResult result;
            auto tokens = result.mutable_tokens();

            ::milvus::proto::milvus::AnalyzerToken token1;
            token1.set_token("dummy");
            token1.set_start_offset(1);
            token1.set_end_offset(5);
            token1.set_position(1);
            token1.set_position_length(4);
            token1.set_hash(888);
            tokens->Add(std::move(token1));

            ::milvus::proto::milvus::AnalyzerToken token2;
            token2.set_token("text");
            token2.set_start_offset(6);
            token2.set_end_offset(10);
            token2.set_position(6);
            token2.set_position_length(3);
            token2.set_hash(999);
            tokens->Add(std::move(token2));

            auto results = response->mutable_results();
            results->Add(std::move(result));

            return ::grpc::Status{};
        });

    milvus::RunAnalyzerArguments args;
    args.SetCollectionName(collection_name);
    args.SetDatabaseName(db_name);
    args.SetFieldName(field_name);
    args.AddText(text);
    args.AddAnalyzerName(analyzer_name);
    args.SetAnalyzerParams(params);
    args.WithDetail(with_detail);
    args.WithHash(with_hash);

    milvus::AnalyzerResults results;
    auto status = client_->RunAnalyzer(args, results);
    EXPECT_TRUE(status.IsOk());

    EXPECT_EQ(results.Results().size(), 1);
    milvus::AnalyzerResult result = results.Results().at(0);
    EXPECT_EQ(result.Tokens().size(), 2);

    milvus::AnalyzerToken token1 = result.Tokens().at(0);
    EXPECT_EQ(token1.token_, "dummy");
    EXPECT_EQ(token1.start_offset_, 1);
    EXPECT_EQ(token1.end_offset_, 5);
    EXPECT_EQ(token1.position_, 1);
    EXPECT_EQ(token1.position_length_, 4);
    EXPECT_EQ(token1.hash_, 888);

    milvus::AnalyzerToken token2 = result.Tokens().at(1);
    EXPECT_EQ(token2.token_, "text");
    EXPECT_EQ(token2.start_offset_, 6);
    EXPECT_EQ(token2.end_offset_, 10);
    EXPECT_EQ(token2.position_, 6);
    EXPECT_EQ(token2.position_length_, 3);
    EXPECT_EQ(token2.hash_, 999);
}
