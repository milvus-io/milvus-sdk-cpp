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

/**
 * @brief Base class of a result highlighter for search.
 *
 * A highlighter serializes its configuration into a parameter map that is sent with the search request.
 */
class MILVUS_SDK_API Highlighter {
 public:
    virtual ~Highlighter() = default;

    /**
     * @brief Get the highlighter type, e.g. "Lexical" or "Semantic".
     * @return the highlight type.
     */
    virtual const std::string&
    HighlightType() const = 0;

    /**
     * @brief Get the serialized configuration parameters of the highlighter.
     * @return the params.
     */
    virtual const std::unordered_map<std::string, std::string>&
    Params() const;

 protected:
    /**
     * @brief Set a serialized configuration parameter.
     * @param [in] key the key.
     * @param [in] value the value.
     */
    void
    SetParam(const std::string& key, std::string value);  // NOLINT(readability-identifier-naming)

    std::unordered_map<std::string, std::string> params_;
};

using HighlighterPtr = std::shared_ptr<Highlighter>;

/**
 * @brief Lexical (BM25-style) highlighter that highlights matching query terms in the retrieved text fields.
 */
class MILVUS_SDK_API LexicalHighlighter : public Highlighter {
 public:
    /**
     * @brief A lexical highlight query: match `text` of the given `type` in a `field`.
     */
    struct HighlightQuery {
        /** @brief query type, e.g. "term" or "phrase". */
        std::string type;
        /** @brief field to highlight. */
        std::string field;
        /** @brief text to match. */
        std::string text;
    };

    /**
     * @brief Get the highlighter type ("Lexical").
     * @return the highlight type.
     */
    const std::string&
    HighlightType() const override;

    /**
     * @brief Set the highlight queries.
     * @param [in] queries the queries.
     */
    LexicalHighlighter&
    WithHighlightQueries(const std::vector<HighlightQuery>& queries);

    /**
     * @brief Add a highlight query.
     * @param [in] query the query.
     */
    LexicalHighlighter&
    AddHighlightQuery(HighlightQuery query);

    /**
     * @brief Add a highlight query by its parts.
     * @param [in] type the type.
     * @param [in] field the field.
     * @param [in] text the text.
     */
    LexicalHighlighter&
    AddHighlightQuery(std::string type, std::string field, std::string text);

    /**
     * @brief Set whether the search input text is highlighted as well.
     * @param [in] value the value.
     */
    LexicalHighlighter&
    WithHighlightSearchText(bool value);

    /**
     * @brief Set the pre-tags wrapping highlighted fragments.
     * @param [in] tags the tags.
     */
    LexicalHighlighter&
    WithPreTags(const std::vector<std::string>& tags);

    /**
     * @brief Add a pre-tag wrapping highlighted fragments.
     * @param [in] tag the tag.
     */
    LexicalHighlighter&
    AddPreTag(std::string tag);

    /**
     * @brief Set the post-tags wrapping highlighted fragments.
     * @param [in] tags the tags.
     */
    LexicalHighlighter&
    WithPostTags(const std::vector<std::string>& tags);

    /**
     * @brief Add a post-tag wrapping highlighted fragments.
     * @param [in] tag the tag.
     */
    LexicalHighlighter&
    AddPostTag(std::string tag);

    /**
     * @brief Set the fragment offset.
     * @param [in] value the value.
     */
    LexicalHighlighter&
    WithFragmentOffset(int64_t value);

    /**
     * @brief Set the fragment size in characters.
     * @param [in] value the value.
     */
    LexicalHighlighter&
    WithFragmentSize(int64_t value);

    /**
     * @brief Set the number of fragments returned per field.
     * @param [in] value the value.
     */
    LexicalHighlighter&
    WithNumOfFragments(int64_t value);

 private:
    void
    syncHighlightQueries();

    void
    syncPreTags();

    void
    syncPostTags();

    std::vector<HighlightQuery> highlight_queries_;
    std::vector<std::string> pre_tags_;
    std::vector<std::string> post_tags_;
};

/**
 * @brief Semantic highlighter that highlights passages semantically related to the queries.
 */
class MILVUS_SDK_API SemanticHighlighter : public Highlighter {
 public:
    /**
     * @brief Get the highlighter type ("Semantic").
     * @return the highlight type.
     */
    const std::string&
    HighlightType() const override;

    /**
     * @brief Set the highlight queries.
     * @param [in] queries the queries.
     */
    SemanticHighlighter&
    WithQueries(const std::vector<std::string>& queries);

    /**
     * @brief Add a highlight query.
     * @param [in] query the query.
     */
    SemanticHighlighter&
    AddQuery(std::string query);

    /**
     * @brief Set the input fields to highlight.
     * @param [in] input_fields the input fields.
     */
    SemanticHighlighter&
    WithInputFields(const std::vector<std::string>& input_fields);

    /**
     * @brief Add an input field to highlight.
     * @param [in] input_field the input field.
     */
    SemanticHighlighter&
    AddInputField(std::string input_field);

    /**
     * @brief Set the pre-tags wrapping highlighted fragments.
     * @param [in] tags the tags.
     */
    SemanticHighlighter&
    WithPreTags(const std::vector<std::string>& tags);

    /**
     * @brief Add a pre-tag wrapping highlighted fragments.
     * @param [in] tag the tag.
     */
    SemanticHighlighter&
    AddPreTag(std::string tag);

    /**
     * @brief Set the post-tags wrapping highlighted fragments.
     * @param [in] tags the tags.
     */
    SemanticHighlighter&
    WithPostTags(const std::vector<std::string>& tags);

    /**
     * @brief Add a post-tag wrapping highlighted fragments.
     * @param [in] tag the tag.
     */
    SemanticHighlighter&
    AddPostTag(std::string tag);

    /**
     * @brief Set the semantic similarity threshold for highlighting.
     * @param [in] value the value.
     */
    SemanticHighlighter&
    WithThreshold(float value);

    /**
     * @brief Set whether only highlighted content is returned.
     * @param [in] value the value.
     */
    SemanticHighlighter&
    WithHighlightOnly(bool value);

    /**
     * @brief Set the model deployment id used for semantic highlighting.
     * @param [in] value the value.
     */
    SemanticHighlighter&
    WithModelDeploymentID(std::string value);

    /**
     * @brief Set the maximum number of queries sent per batch.
     * @param [in] value the value.
     */
    SemanticHighlighter&
    WithMaxClientBatchSize(int64_t value);

 private:
    void
    syncQueries();

    void
    syncInputFields();

    void
    syncPreTags();

    void
    syncPostTags();

    std::vector<std::string> queries_;
    std::vector<std::string> input_fields_;
    std::vector<std::string> pre_tags_;
    std::vector<std::string> post_tags_;
};

}  // namespace milvus
