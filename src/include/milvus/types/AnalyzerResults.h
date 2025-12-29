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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../Status.h"

namespace milvus {

/**
 * @brief Token details for MilvusClient::RunAnalyzer().
 */
struct AnalyzerToken {
 public:
    std::string token_;
    int64_t start_offset_;
    int64_t end_offset_;
    int64_t position_;
    int64_t position_length_;
    uint32_t hash_;
};

/**
 * @brief Result list for MilvusClient::RunAnalyzer().
 */
class AnalyzerResult {
 public:
    explicit AnalyzerResult(std::vector<AnalyzerToken>&& tokens);

    /**
     * @brief Set tokens to be analyzed.
     */
    const std::vector<AnalyzerToken>&
    Tokens() const;

    /**
     * @brief Add a token to be analyzed.
     */
    Status
    AddToken(AnalyzerToken&& token);

 private:
    std::vector<AnalyzerToken> tokens_;
};

using AnalyzerResults = std::vector<AnalyzerResult>;

}  // namespace milvus
