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

#include "milvus/types/Highlighter.h"

#include <nlohmann/json.hpp>
#include <utility>

namespace milvus {

const std::unordered_map<std::string, std::string>&
Highlighter::Params() const {
    return params_;
}

void
Highlighter::SetParam(const std::string& key, std::string value) {
    params_[key] = std::move(value);
}

const std::string&
LexicalHighlighter::HighlightType() const {
    static const std::string kType = "Lexical";
    return kType;
}

LexicalHighlighter&
LexicalHighlighter::WithHighlightQueries(const std::vector<HighlightQuery>& queries) {
    highlight_queries_ = queries;
    SyncHighlightQueries();
    return *this;
}

LexicalHighlighter&
LexicalHighlighter::AddHighlightQuery(HighlightQuery query) {
    highlight_queries_.emplace_back(std::move(query));
    SyncHighlightQueries();
    return *this;
}

LexicalHighlighter&
LexicalHighlighter::AddHighlightQuery(std::string type, std::string field, std::string text) {
    highlight_queries_.push_back({std::move(type), std::move(field), std::move(text)});
    SyncHighlightQueries();
    return *this;
}

LexicalHighlighter&
LexicalHighlighter::WithHighlightSearchText(bool value) {
    SetParam("highlight_search_text", value ? "true" : "false");
    return *this;
}

LexicalHighlighter&
LexicalHighlighter::WithPreTags(const std::vector<std::string>& tags) {
    pre_tags_ = tags;
    SyncPreTags();
    return *this;
}

LexicalHighlighter&
LexicalHighlighter::AddPreTag(std::string tag) {
    pre_tags_.emplace_back(std::move(tag));
    SyncPreTags();
    return *this;
}

LexicalHighlighter&
LexicalHighlighter::WithPostTags(const std::vector<std::string>& tags) {
    post_tags_ = tags;
    SyncPostTags();
    return *this;
}

LexicalHighlighter&
LexicalHighlighter::AddPostTag(std::string tag) {
    post_tags_.emplace_back(std::move(tag));
    SyncPostTags();
    return *this;
}

LexicalHighlighter&
LexicalHighlighter::WithFragmentOffset(int64_t value) {
    SetParam("fragment_offset", std::to_string(value));
    return *this;
}

LexicalHighlighter&
LexicalHighlighter::WithFragmentSize(int64_t value) {
    SetParam("fragment_size", std::to_string(value));
    return *this;
}

LexicalHighlighter&
LexicalHighlighter::WithNumOfFragments(int64_t value) {
    SetParam("num_of_fragments", std::to_string(value));
    return *this;
}

void
LexicalHighlighter::SyncHighlightQueries() {
    nlohmann::json queries = nlohmann::json::array();
    for (const auto& query : highlight_queries_) {
        queries.push_back({{"type", query.type}, {"field", query.field}, {"text", query.text}});
    }
    SetParam("highlight_query", queries.dump());
}

void
LexicalHighlighter::SyncPreTags() {
    SetParam("pre_tags", nlohmann::json(pre_tags_).dump());
}

void
LexicalHighlighter::SyncPostTags() {
    SetParam("post_tags", nlohmann::json(post_tags_).dump());
}

const std::string&
SemanticHighlighter::HighlightType() const {
    static const std::string kType = "Semantic";
    return kType;
}

SemanticHighlighter&
SemanticHighlighter::WithQueries(const std::vector<std::string>& queries) {
    queries_ = queries;
    SyncQueries();
    return *this;
}

SemanticHighlighter&
SemanticHighlighter::AddQuery(std::string query) {
    queries_.emplace_back(std::move(query));
    SyncQueries();
    return *this;
}

SemanticHighlighter&
SemanticHighlighter::WithInputFields(const std::vector<std::string>& input_fields) {
    input_fields_ = input_fields;
    SyncInputFields();
    return *this;
}

SemanticHighlighter&
SemanticHighlighter::AddInputField(std::string input_field) {
    input_fields_.emplace_back(std::move(input_field));
    SyncInputFields();
    return *this;
}

SemanticHighlighter&
SemanticHighlighter::WithPreTags(const std::vector<std::string>& tags) {
    pre_tags_ = tags;
    SyncPreTags();
    return *this;
}

SemanticHighlighter&
SemanticHighlighter::AddPreTag(std::string tag) {
    pre_tags_.emplace_back(std::move(tag));
    SyncPreTags();
    return *this;
}

SemanticHighlighter&
SemanticHighlighter::WithPostTags(const std::vector<std::string>& tags) {
    post_tags_ = tags;
    SyncPostTags();
    return *this;
}

SemanticHighlighter&
SemanticHighlighter::AddPostTag(std::string tag) {
    post_tags_.emplace_back(std::move(tag));
    SyncPostTags();
    return *this;
}

SemanticHighlighter&
SemanticHighlighter::WithThreshold(float value) {
    SetParam("threshold", std::to_string(value));
    return *this;
}

SemanticHighlighter&
SemanticHighlighter::WithHighlightOnly(bool value) {
    SetParam("highlight_only", value ? "true" : "false");
    return *this;
}

SemanticHighlighter&
SemanticHighlighter::WithModelDeploymentID(std::string value) {
    SetParam("model_deployment_id", std::move(value));
    return *this;
}

SemanticHighlighter&
SemanticHighlighter::WithMaxClientBatchSize(int64_t value) {
    SetParam("max_client_batch_size", std::to_string(value));
    return *this;
}

void
SemanticHighlighter::SyncQueries() {
    SetParam("queries", nlohmann::json(queries_).dump());
}

void
SemanticHighlighter::SyncInputFields() {
    SetParam("input_fields", nlohmann::json(input_fields_).dump());
}

void
SemanticHighlighter::SyncPreTags() {
    SetParam("pre_tags", nlohmann::json(pre_tags_).dump());
}

void
SemanticHighlighter::SyncPostTags() {
    SetParam("post_tags", nlohmann::json(post_tags_).dump());
}

}  // namespace milvus
