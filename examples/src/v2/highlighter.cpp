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

#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

/**
 * Demonstrates lexical highlighter usage for BM25 search results.
 *
 * Prerequisites:
 * - A running Milvus instance on localhost:19530
 * - Server-side text search and highlighter support enabled
 */
namespace {
const char* const collection_name = "java_sdk_example_highlighter_v2";
const char* const field_id = "id";
const char* const field_title = "title";
const char* const field_vector = "vector";
const char* const field_text = "text";

std::string
JoinQueryTexts(const std::vector<std::string>& query_texts) {
    std::string joined;
    for (size_t i = 0; i < query_texts.size(); ++i) {
        if (i != 0) {
            joined += " ";
        }
        joined += query_texts.at(i);
    }
    return joined;
}

std::string
FormatQueryTextsForDisplay(const std::vector<std::string>& query_texts) {
    std::string formatted = "[";
    for (size_t i = 0; i < query_texts.size(); ++i) {
        if (i != 0) {
            formatted += ", ";
        }
        formatted += query_texts.at(i);
    }
    formatted += "]";
    return formatted;
}

void
createCollection(milvus::MilvusClientV2Ptr& client) {
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, false});
    collection_schema->AddField(milvus::FieldSchema(field_title, milvus::DataType::VARCHAR)
                                    .WithMaxLength(512)
                                    .EnableAnalyzer(true)
                                    .EnableMatch(true));
    collection_schema->AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR)
                                    .WithMaxLength(65535)
                                    .EnableAnalyzer(true)
                                    .EnableMatch(true));
    collection_schema->AddField(milvus::FieldSchema(field_vector, milvus::DataType::SPARSE_FLOAT_VECTOR));

    milvus::FunctionPtr function = std::make_shared<milvus::Function>("function_bm25", milvus::FunctionType::BM25);
    function->AddInputFieldName(field_title);
    function->AddOutputFieldName(field_vector);
    collection_schema->AddFunction(function);

    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::SPARSE_INVERTED_INDEX,
                                   milvus::MetricType::BM25);
    status = client->CreateCollection(milvus::CreateCollectionRequest()
                                          .WithCollectionName(collection_name)
                                          .WithCollectionSchema(collection_schema)
                                          .AddIndex(std::move(index_vector))
                                          .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED));
    util::CheckStatus(std::string("create collection: ") + collection_name, status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus(std::string("load collection: ") + collection_name, status);
    std::cout << "Collection created: " << collection_name << std::endl;
}

void
insertData(milvus::MilvusClientV2Ptr& client) {
    milvus::EntityRows rows;

    milvus::EntityRow row0;
    row0[field_id] = 0;
    row0[field_title] = "Milvus for scale";
    row0[field_text] =
        "Milvus is an open-source vector database built for scale. This paragraph is intentionally long so the keyword "
        "search appears much later in the same text fragment. Search is a core capability for information retrieval "
        "systems.";
    rows.emplace_back(std::move(row0));

    milvus::EntityRow row1;
    row1[field_id] = 1;
    row1[field_title] = "Full text search";
    row1[field_text] =
        "Milvus supports full text search with analyzers and BM25. This sentence adds enough spacing and extra wording "
        "to separate the two highlighted terms into different regions for the lexical highlighter example.";
    rows.emplace_back(std::move(row1));

    milvus::EntityRow row2;
    row2[field_id] = 2;
    row2[field_title] = "RAG systems";
    row2[field_text] = "Vector databases help retrieval augmented generation systems.";
    rows.emplace_back(std::move(row2));

    milvus::EntityRow row3;
    row3[field_id] = 3;
    row3[field_title] = "Milvus users";
    row3[field_text] =
        "This example demonstrates highlighted snippets for modern applications. The word search is placed here with a "
        "lot of filler text before Milvus appears again near the end of the document to encourage multiple fragments "
        "in highlighter output for Milvus users.";
    rows.emplace_back(std::move(row3));

    milvus::InsertResponse resp_insert;
    auto status = client->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), resp_insert);
    util::CheckStatus("insert", status);
    std::cout << "Inserted " << resp_insert.Results().InsertCount() << " rows" << std::endl;

    status = client->Flush(milvus::FlushRequest().AddCollectionName(collection_name));
    util::CheckStatus("flush collection", status);
}

void
PrintHighlightResult(const milvus::HighlightResults& highlight_results, const std::string& field_name) {
    auto iter = highlight_results.find(field_name);
    if (iter == highlight_results.end()) {
        return;
    }

    const auto& highlight_result = iter->second;
    std::cout << "  highlighted field: " << highlight_result.field_name << std::endl;
    std::cout << "  fragments: " << nlohmann::json(highlight_result.fragments).dump() << std::endl;
    std::cout << "  scores: " << nlohmann::json(highlight_result.scores).dump() << std::endl;
}

void
searchWithHighlighter(milvus::MilvusClientV2Ptr& client, const std::vector<std::string>& query_texts) {
    auto highlighter = std::make_shared<milvus::LexicalHighlighter>();
    highlighter->AddPreTag("<em>").AddPostTag("</em>").WithFragmentSize(40).WithNumOfFragments(10);
    for (const auto& query_text : query_texts) {
        highlighter->AddHighlightQuery("TextMatch", field_text, query_text);
    }

    auto request =
        milvus::SearchRequest()
            .WithCollectionName(collection_name)
            .WithAnnsField(field_vector)
            .WithFilter(std::string("TEXT_MATCH(") + field_text + ", \"" + JoinQueryTexts(query_texts) + "\")")
            .WithLimit(3)
            .WithMetricType(milvus::MetricType::BM25)
            .AddOutputField(field_title)
            .AddOutputField(field_text)
            .WithHighlighter(highlighter);
    for (const auto& query_text : query_texts) {
        request.AddEmbeddedText(query_text);
    }

    milvus::SearchResponse response;
    auto status = client->Search(request, response);
    util::CheckStatus("search", status);

    std::cout << "\nSearch with lexical highlighter: " << FormatQueryTextsForDisplay(query_texts) << std::endl;
    for (const auto& result : response.Results().Results()) {
        std::cout << "\n-----------------------------------------------------------------------------" << std::endl;
        for (int i = 0; i < static_cast<int>(result.GetRowCount()); ++i) {
            milvus::EntityRow row;
            status = result.OutputRow(i, row);
            util::CheckStatus("get output row", status);
            std::cout << row << std::endl;

            milvus::HighlightResults highlight_results;
            status = result.OutputHighlightResult(i, highlight_results);
            util::CheckStatus("get highlight result", status);
            PrintHighlightResult(highlight_results, field_text);
            PrintHighlightResult(highlight_results, field_title);
            std::cout << "  title: " << row[field_title] << std::endl;
            std::cout << "  text: " << row[field_text] << std::endl;
        }
    }
    std::cout << "=============================================================" << std::endl;
}

void
runExample(milvus::MilvusClientV2Ptr& client) {
    createCollection(client);
    insertData(client);
    searchWithHighlighter(client, {"milvus users", "text search"});
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    runExample(client);

    client->Disconnect();
    return 0;
}
