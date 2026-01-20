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

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {
const char* const collection_name = "CPP_V2_MULTI_ANALYZER";
const char* const field_id = "id";
const char* const field_vector = "vector";
const char* const field_text = "text";
const char* const field_language = "language";

void
buildCollection(milvus::MilvusClientV2Ptr& client) {
    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, true});
    collection_schema->AddField(milvus::FieldSchema(field_vector, milvus::DataType::SPARSE_FLOAT_VECTOR));

    // apply multiple analyzers to the text field, so that insert data can specify different tokenizers for each row.
    // in this example, texts are written by multiple languages, so we use multiple analyzers to handle different texts.
    // to use multiple analyzers, there must be a field to specify the language type, in this example, the "language"
    // field is used for this purpose. multiple analyzers is optional, no need to set it if the data only contains one
    // language, no need to add the "language" field if the data only contains one language.
    // tokenizer:
    //  english: https://milvus.io/docs/english-analyzer.md
    //  chinese: https://milvus.io/docs/chinese-analyzer.md
    //  lindera: https://milvus.io/docs/lindera-tokenizer.md
    //  icu: https://milvus.io/docs/icu-tokenizer.md
    // filter:
    //  lowercase: https://milvus.io/docs/lowercase-filter.md
    //  removepunct: https://milvus.io/docs/removepunct-filter.md
    //  asciifolding: https://milvus.io/docs/ascii-folding-filter.md
    nlohmann::json multi_analyzers = {
        {"analyzers",
         {{"english", {{"type", "english"}}},
          {"chinese", {{"tokenizer", "jieba"}, {"filter", {"lowercase", "removepunct"}}}},
          {"japanese", {{"tokenizer", {{"type", "lindera"}, {"dict_kind", "ipadic"}}}}},
          {"default", {{"tokenizer", "icu"}, {"filter", {"lowercase", "removepunct", "asciifolding"}}}}}},
        {"by_field", field_language},
        {"alias", {{"cn", "chinese"}, {"en", "english"}, {"jap", "japanese"}}}};
    collection_schema->AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR)
                                    .WithMaxLength(65535)
                                    .EnableAnalyzer(true)
                                    .WithMultiAnalyzerParams(multi_analyzers));
    collection_schema->AddField(milvus::FieldSchema(field_language, milvus::DataType::VARCHAR).WithMaxLength(100));

    // define the BM25 function, milvus will automatically generate sparse vector by BM25 algorithm for the "text" field
    // the sparse vectors are stored in the "vector" field, and are invisible to users
    milvus::FunctionPtr function = std::make_shared<milvus::Function>("function_bm25", milvus::FunctionType::BM25);
    function->AddInputFieldName(field_text);
    function->AddOutputFieldName(field_vector);
    collection_schema->AddFunction(function);

    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus(std::string("create collection: ") + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::SPARSE_INVERTED_INDEX,
                                   milvus::MetricType::BM25);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus(std::string("load collection: ") + collection_name, status);

    // insert some rows by row-based
    const std::vector<std::string> english_content = {
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

    const std::vector<std::string> chinese_content = {
        "人工智能正在改变技术领域",
        "机器学习模型需要大型数据集",
        "Milvus 是一个高性能、可扩展的向量数据库！",
    };

    const std::vector<std::string> japanese_content = {
        "Milvusの新機能をご確認くださいこのページでは",
        "非構造化データやマルチモーダルデータを構造化されたコレクションに整理することができます",
        "主な利点はデータアクセスパターンにある",
    };

    const std::vector<std::string> mix_content = {
        "토큰화 도구는 소프트웨어 국제화를 위한 핵심 도구를 제공하는",
        "Les applications qui suivent le temps à travers les régions",
        "Sin embargo, esto puede aumentar la complejidad de las consultas y de la gestión",
        "المثال، يوضح الرمز التالي كيفية إضافة عامل تصفية الحقل القياسي إلى بحث متجه",
    };

    milvus::EntityRows rows;
    rows.reserve(english_content.size() + chinese_content.size() + japanese_content.size() + mix_content.size());
    for (const auto& text : english_content) {
        milvus::EntityRow row;
        row[field_text] = text;
        row[field_language] =
            "en";  // this alias is defined in the multi_analyzers, both alias and original name are ok here
        rows.emplace_back(std::move(row));
    }

    for (const auto& text : chinese_content) {
        milvus::EntityRow row;
        row[field_text] = text;
        row[field_language] =
            "cn";  // this alias is defined in the multi_analyzers, both alias and original name are ok here
        rows.emplace_back(std::move(row));
    }

    for (const auto& text : japanese_content) {
        milvus::EntityRow row;
        row[field_text] = text;
        row[field_language] =
            "jap";  // this alias is defined in the multi_analyzers, both alias and original name are ok here
        rows.emplace_back(std::move(row));
    }

    for (const auto& text : mix_content) {
        milvus::EntityRow row;
        row[field_text] = text;
        row[field_language] = "default";
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse resp_insert;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            resp_insert);
    util::CheckStatus("insert", status);

    // get row count
    auto request = milvus::QueryRequest()
                       .WithCollectionName(collection_name)
                       .AddOutputField("count(*)")
                       .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse response;
    status = client->Query(request, response);
    util::CheckStatus("query count(*)", status);
    std::cout << "count(*) = " << response.Results().GetRowCount() << std::endl;
}

void
searchByText(milvus::MilvusClientV2Ptr& client, std::string text, std::string language) {
    std::cout << "============================== " << language << " =================================" << std::endl;
    std::cout << "Search by text: " << text << std::endl;

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .AddEmbeddedText(text)
                       .WithLimit(5)
                       .WithAnnsField(field_vector)
                       .AddOutputField(field_text)
                       .AddOutputField(field_language)
                       .AddExtraParam("analyzer_name", language)  // choose a tokenizer to split the text
                       .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResponse response;
    auto status = client->Search(request, response);
    util::CheckStatus("search", status);

    auto& result = response.Results().Results().at(0);
    milvus::EntityRows output_rows;
    status = result.OutputRows(output_rows);
    util::CheckStatus("get output rows", status);
    for (const auto& row : output_rows) {
        std::cout << "\t" << row << std::endl;
    }
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    buildCollection(client);

    searchByText(client, "Milvus vector database", "english");
    searchByText(client, "人工智能与机器学习", "chinese");
    searchByText(client, "非構造化データ", "japanese");
    searchByText(client, "Gestion des applications", "default");

    client->Disconnect();
    return 0;
}
