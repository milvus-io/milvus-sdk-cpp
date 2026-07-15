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

#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {
const char* const collection_name = "cpp_sdk_example_minhash_v2";
const char* const id_field = "id";
const char* const text_field = "text";
const char* const signature_field = "minhash_signature";
constexpr int64_t num_hashes = 16;
constexpr int64_t signature_dimension = num_hashes * 32;

void
createDedupCollection(milvus::MilvusClientV2Ptr& client) {
    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));

    auto schema = std::make_shared<milvus::CollectionSchema>();
    schema->SetEnableDynamicField(true);
    schema->AddField({id_field, milvus::DataType::INT64, "", true, false});
    schema->AddField(milvus::FieldSchema(text_field, milvus::DataType::VARCHAR).WithMaxLength(65535));
    schema->AddField(
        milvus::FieldSchema(signature_field, milvus::DataType::BINARY_VECTOR).WithDimension(signature_dimension));

    auto function = std::make_shared<milvus::Function>("text_to_minhash", milvus::FunctionType::MINHASH);
    function->AddInputFieldName(text_field);
    function->AddOutputFieldName(signature_field);
    function->AddParam("num_hashes", std::to_string(num_hashes));
    function->AddParam("shingle_size", "3");
    function->AddParam("token_level", "word");
    schema->AddFunction(function);

    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
    util::CheckStatus(std::string("create collection: ") + collection_name, status);

    milvus::IndexDesc index(signature_field, "", milvus::IndexType::MINHASH_LSH, milvus::MetricType::MHJACCARD);
    index.AddExtraParam("mh_lsh_band", "8");
    index.AddExtraParam("with_raw_data", "true");
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index)));
    util::CheckStatus("create index on MinHash signature field", status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus(std::string("load collection: ") + collection_name, status);
    std::cout << "Collection created" << std::endl;
}

std::vector<std::string>
insertTexts(milvus::MilvusClientV2Ptr& client) {
    std::vector<std::string> texts = {
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown fox jumped over a lazy dog.",
        "The fast brown fox leaps over the sleepy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Deep learning transforms artificial intelligence research.",
        "Completely unrelated text about cooking recipes.",
        "Completely unrelated text about cooking recipes!",
    };

    milvus::EntityRows rows;
    for (size_t i = 0; i < texts.size(); ++i) {
        milvus::EntityRow row;
        row[id_field] = static_cast<int64_t>(i + 1);
        row[text_field] = texts.at(i);
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse insert_response;
    auto status = client->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), insert_response);
    util::CheckStatus("insert texts", status);

    auto request = milvus::QueryRequest()
                       .WithCollectionName(collection_name)
                       .AddOutputField("count(*)")
                       .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    milvus::QueryResponse query_response;
    status = client->Query(request, query_response);
    util::CheckStatus("query count(*)", status);

    milvus::EntityRows count_rows;
    status = query_response.Results().OutputRows(count_rows);
    util::CheckStatus("read count(*)", status);
    std::cout << count_rows.at(0).at("count(*)") << " rows in collection" << std::endl;
    return texts;
}

milvus::SearchResponse
searchByText(milvus::MilvusClientV2Ptr& client, const std::string& text, int64_t top_k) {
    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithAnnsField(signature_field)
                       .AddEmbeddedText(text)
                       .WithMetricType(milvus::MetricType::MHJACCARD)
                       .AddExtraParam("mh_search_with_jaccard", "true")
                       .AddExtraParam("refine_k", "50")
                       .WithLimit(top_k)
                       .AddOutputField(id_field)
                       .AddOutputField(text_field)
                       .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResponse response;
    auto status = client->Search(request, response);
    util::CheckStatus("search by text", status);
    return response;
}

void
deduplicateTexts(milvus::MilvusClientV2Ptr& client, const std::vector<std::string>& texts, float similarity_threshold,
                 int64_t top_k) {
    std::set<int64_t> unique_ids;
    std::vector<std::string> duplicates;

    for (size_t i = 0; i < texts.size(); ++i) {
        const auto document_id = static_cast<int64_t>(i + 1);
        auto response = searchByText(client, texts.at(i), top_k);
        const auto& result = response.Results().Results().at(0);

        milvus::EntityRows output_rows;
        auto status = result.OutputRows(output_rows);
        util::CheckStatus("get search output rows", status);

        bool is_duplicate = false;
        for (size_t hit_index = 0; hit_index < output_rows.size(); ++hit_index) {
            const auto hit_id = output_rows.at(hit_index).at(id_field).get<int64_t>();
            if (hit_id == document_id) {
                continue;
            }

            const auto score = result.Scores().at(hit_index);
            if (score >= similarity_threshold && hit_id < document_id) {
                std::ostringstream message;
                message << "ID " << document_id << " is duplicate of ID " << hit_id << ", similarity=" << std::fixed
                        << std::setprecision(4) << score;
                duplicates.emplace_back(message.str());
                is_duplicate = true;
                break;
            }
        }

        if (!is_duplicate) {
            unique_ids.insert(document_id);
        }
    }

    std::cout << "\nUnique texts:" << std::endl;
    for (size_t i = 0; i < texts.size(); ++i) {
        if (unique_ids.count(static_cast<int64_t>(i + 1)) > 0) {
            std::cout << "  - " << texts.at(i) << std::endl;
        }
    }

    std::cout << "\nDuplicates:" << std::endl;
    if (duplicates.empty()) {
        std::cout << "  (none)" << std::endl;
    } else {
        for (const auto& duplicate : duplicates) {
            std::cout << "  - " << duplicate << std::endl;
        }
    }
}

}  // namespace

int
main(int argc, char* argv[]) {
    std::cout << "Example start..." << std::endl;

    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    createDedupCollection(client);
    auto texts = insertTexts(client);
    deduplicateTexts(client, texts, 0.8F, 5);

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus(std::string("drop collection: ") + collection_name, status);
    client->Disconnect();
    return 0;
}
