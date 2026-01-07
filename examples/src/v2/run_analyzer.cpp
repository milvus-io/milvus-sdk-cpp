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

#include <iostream>
#include <string>
#include <thread>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {
void
printAnalyzerResults(const milvus::AnalyzerResults& results) {
    for (const auto& result : results) {
        std::cout << "\t------------------------------" << std::endl;
        for (const auto& token : result.Tokens()) {
            std::cout << "\t{token: " << token.token_ << ", start: " << token.start_offset_
                      << ", end: " << token.end_offset_ << ", position: " << token.position_
                      << ", position_len: " << token.position_length_ << ", hash: " << token.hash_ << "}" << std::endl;
        }
        std::cout << "\t------------------------------" << std::endl;
    }
}

void
runAnalyzer(milvus::MilvusClientV2Ptr& client, const nlohmann::json& analyzer_params, const std::string& text) {
    std::cout << "\nRun analyzer params: " << analyzer_params.dump() << std::endl;
    std::cout << "Text: " << text << std::endl;
    auto request =
        milvus::RunAnalyzerRequest().AddText(text).WithAnalyzerParams(analyzer_params).WithDetail(true).WithHash(true);

    milvus::RunAnalyzerResponse response;
    auto status = client->RunAnalyzer(request, response);
    util::CheckStatus("run analyzer", status);
    printAnalyzerResults(response.Results());
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    // stop
    {
        nlohmann::json analyzer_params = {
            {"tokenizer", "standard"},
            {"filter", {{{"type", "stop"}, {"stop_words", {"and", "for"}}}}},
        };
        std::string text = "Milvus supports L2 distance and IP similarity for float vector.";
        runAnalyzer(client, analyzer_params, text);
    }

    // jieba
    {
        nlohmann::json analyzer_params = {{"tokenizer", "jieba"}, {"filter", {"cnalphanumonly"}}};
        std::string text = "Milvus 是 LF AI & Data Foundation 下的一个开源项目，以 Apache 2.0 许可发布。";
        runAnalyzer(client, analyzer_params, text);
    }

    // lindera
    {
        nlohmann::json analyzer_params = {{"tokenizer", {{"type", "lindera"}, {"dict_kind", "ipadic"}}}};
        std::string text = "東京スカイツリーの最寄り駅はとうきょうスカイツリー駅で";
        runAnalyzer(client, analyzer_params, text);
    }

    // icu
    {
        nlohmann::json analyzer_params = {{"tokenizer", "icu"}};
        std::string text = "Привет! Как дела?";
        runAnalyzer(client, analyzer_params, text);
    }

    // length
    {
        nlohmann::json analyzer_params = {{"tokenizer", "standard"}, {"filter", {{{"type", "length"}, {"max", 6}}}}};
        std::string text = "The length filter allows control over token length requirements for text processing.";
        runAnalyzer(client, analyzer_params, text);
    }

    // decompounder
    {
        nlohmann::json analyzer_params = {
            {"tokenizer", "standard"},
            {"filter",
             {{{"type", "decompounder"}, {"word_list", {"dampf", "schiff", "fahrt", "brot", "backen", "automat"}}}}}};
        std::string text = "dampfschifffahrt brotbackautomat";
        runAnalyzer(client, analyzer_params, text);
    }

    // stemmer
    {
        nlohmann::json analyzer_params = {{"tokenizer", "standard"},
                                          {"filter", {{{"type", "stemmer"}, {"language", "english"}}}}};
        std::string text = "running runs looked ran runner";
        runAnalyzer(client, analyzer_params, text);
    }

    // regex
    {
        nlohmann::json analyzer_params = {{"tokenizer", "standard"},
                                          {"filter", {{{"type", "regex"}, {"expr", "^(?!test)"}}}}};
        std::string text = "testItem apple testCase banana";
        runAnalyzer(client, analyzer_params, text);
    }

    client->Disconnect();
    return 0;
}
