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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "milvus/Export.h"

namespace milvus {

class MILVUS_SDK_API Highlighter {
 public:
    virtual ~Highlighter() = default;

    virtual const std::string&
    HighlightType() const = 0;

    virtual const std::unordered_map<std::string, std::string>&
    Params() const;

 protected:
    void
    SetParam(const std::string& key, std::string value);

    std::unordered_map<std::string, std::string> params_;
};

using HighlighterPtr = std::shared_ptr<Highlighter>;

class MILVUS_SDK_API LexicalHighlighter : public Highlighter {
 public:
    struct HighlightQuery {
        std::string type;
        std::string field;
        std::string text;
    };

    const std::string&
    HighlightType() const override;

    LexicalHighlighter&
    WithHighlightQueries(const std::vector<HighlightQuery>& queries);

    LexicalHighlighter&
    AddHighlightQuery(HighlightQuery query);

    LexicalHighlighter&
    AddHighlightQuery(std::string type, std::string field, std::string text);

    LexicalHighlighter&
    WithHighlightSearchText(bool value);

    LexicalHighlighter&
    WithPreTags(const std::vector<std::string>& tags);

    LexicalHighlighter&
    AddPreTag(std::string tag);

    LexicalHighlighter&
    WithPostTags(const std::vector<std::string>& tags);

    LexicalHighlighter&
    AddPostTag(std::string tag);

    LexicalHighlighter&
    WithFragmentOffset(int64_t value);

    LexicalHighlighter&
    WithFragmentSize(int64_t value);

    LexicalHighlighter&
    WithNumOfFragments(int64_t value);

 private:
    void
    SyncHighlightQueries();

    void
    SyncPreTags();

    void
    SyncPostTags();

    std::vector<HighlightQuery> highlight_queries_;
    std::vector<std::string> pre_tags_;
    std::vector<std::string> post_tags_;
};

class MILVUS_SDK_API SemanticHighlighter : public Highlighter {
 public:
    const std::string&
    HighlightType() const override;

    SemanticHighlighter&
    WithQueries(const std::vector<std::string>& queries);

    SemanticHighlighter&
    AddQuery(std::string query);

    SemanticHighlighter&
    WithInputFields(const std::vector<std::string>& input_fields);

    SemanticHighlighter&
    AddInputField(std::string input_field);

    SemanticHighlighter&
    WithPreTags(const std::vector<std::string>& tags);

    SemanticHighlighter&
    AddPreTag(std::string tag);

    SemanticHighlighter&
    WithPostTags(const std::vector<std::string>& tags);

    SemanticHighlighter&
    AddPostTag(std::string tag);

    SemanticHighlighter&
    WithThreshold(float value);

    SemanticHighlighter&
    WithHighlightOnly(bool value);

    SemanticHighlighter&
    WithModelDeploymentID(std::string value);

    SemanticHighlighter&
    WithMaxClientBatchSize(int64_t value);

 private:
    void
    SyncQueries();

    void
    SyncInputFields();

    void
    SyncPreTags();

    void
    SyncPostTags();

    std::vector<std::string> queries_;
    std::vector<std::string> input_fields_;
    std::vector<std::string> pre_tags_;
    std::vector<std::string> post_tags_;
};

}  // namespace milvus
