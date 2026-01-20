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
runAnalyzer(milvus::MilvusClientV2Ptr& client, const nlohmann::json& analyzer_params) {
    std::cout << "Run analyzer params: " << analyzer_params.dump() << std::endl;

    const std::vector<std::string> text_content = {
        "Milvus is an open-source vector database",
        "AI applications help people better life",
        "Will the electric car replace gas-powered car?",
        "LangChain is a composable framework to build with LLMs. Milvus is integrated into LangChain.",
        "RAG is the process of optimizing the output of a large language model",
        "Newton is one of the greatest scientist of human history",
        "Metric type L2 is Euclidean distance",
        "Embeddings represent real-world objects, like words, images, or videos, in a form that computers can process.",
        "The moon is 384,400 km distance away from earth",
        "Milvus supports L2 distance and IP similarity for float vector.",
    };

    auto request = milvus::RunAnalyzerRequest()
                       .WithTexts(text_content)
                       .WithAnalyzerParams(analyzer_params)
                       .WithDetail(true)
                       .WithHash(true);

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

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    nlohmann::json params_1 = {
        {"tokenizer", "standard"},
        {"filter", {{{"type", "stop"}, {"stop_words", {"of"}}}}},
    };
    runAnalyzer(client, params_1);

    nlohmann::json params_2 = {
        {"tokenizer", "standard"},
        {"filter", {{{"type", "stop"}, {"stop_words", {"is", "of", "for"}}}}},
    };
    runAnalyzer(client, params_2);

    client->Disconnect();
    return 0;
}
