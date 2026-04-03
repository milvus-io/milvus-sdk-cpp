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

#include <set>

#include "MilvusServerTest.h"

// test data shared by all test cases
static const std::vector<std::string> kTextContent = {
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

///////////////////////////////////////////////////////////////////////////////
// TEXT_MATCH tests: varchar field with analyzer + EnableMatch
///////////////////////////////////////////////////////////////////////////////
class MilvusServerTestTextMatch : public ::testing::Test {
 protected:
    static std::shared_ptr<milvus::MilvusClientV2> client_;
    static std::string collection_name;
    static constexpr uint32_t dimension = 4;

    static void
    SetUpTestSuite() {
        const char* host = std::getenv("MILVUS_HOST");
        milvus::ConnectParam connect_param{host ? host : "localhost", 19530};
        client_ = milvus::MilvusClientV2::Create();
        auto status = client_->Connect(connect_param);
        milvus::test::ExpectStatusOK(status);

        collection_name = milvus::test::RanName("TextMatch_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, false));
        schema->AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(dimension));
        schema->AddField(milvus::FieldSchema("text", milvus::DataType::VARCHAR, "text")
                             .WithMaxLength(1024)
                             .EnableAnalyzer(true)
                             .EnableMatch(true));

        status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        milvus::IndexDesc idx("vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
        status = client_->CreateIndex(
            milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(idx)));
        milvus::test::ExpectStatusOK(status);

        // insert text data
        milvus::EntityRows rows_data;
        for (size_t i = 0; i < kTextContent.size(); ++i) {
            nlohmann::json row;
            row["id"] = static_cast<int64_t>(i);
            row["vec"] = std::vector<float>{0.1f * (i + 1), 0.2f, 0.3f, 0.4f};
            row["text"] = kTextContent[i];
            rows_data.emplace_back(std::move(row));
        }

        milvus::InsertRequest insert_req;
        insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
        milvus::InsertResponse insert_resp;
        status = client_->Insert(insert_req, insert_resp);
        milvus::test::ExpectStatusOK(status);

        status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
        milvus::test::ExpectStatusOK(status);
    }

    static void
    TearDownTestSuite() {
        if (client_) {
            client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
            client_->Disconnect();
            client_.reset();
        }
    }
};

std::shared_ptr<milvus::MilvusClientV2> MilvusServerTestTextMatch::client_;
std::string MilvusServerTestTextMatch::collection_name;

TEST_F(MilvusServerTestTextMatch, QueryWithTextMatch) {
    milvus::QueryRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter(R"(TEXT_MATCH(text, "distance"))");
    req.AddOutputField("text");
    req.WithLimit(100);
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::QueryResponse resp;
    auto status = client_->Query(req, resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    resp.Results().OutputRows(rows);
    EXPECT_GE(rows.size(), 2);
    for (const auto& row : rows) {
        auto text = row["text"].get<std::string>();
        EXPECT_NE(text.find("distance"), std::string::npos) << "expected 'distance' in: " << text;
    }
}

TEST_F(MilvusServerTestTextMatch, QueryWithTextMatchOr) {
    milvus::QueryRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter(R"(TEXT_MATCH(text, "Milvus") or TEXT_MATCH(text, "distance"))");
    req.AddOutputField("text");
    req.WithLimit(100);
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::QueryResponse resp;
    auto status = client_->Query(req, resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    resp.Results().OutputRows(rows);
    EXPECT_GE(rows.size(), 4);
    for (const auto& row : rows) {
        auto text = row["text"].get<std::string>();
        bool has_milvus = text.find("Milvus") != std::string::npos || text.find("milvus") != std::string::npos;
        bool has_distance = text.find("distance") != std::string::npos;
        EXPECT_TRUE(has_milvus || has_distance) << "expected 'Milvus' or 'distance' in: " << text;
    }
}

TEST_F(MilvusServerTestTextMatch, QueryWithTextMatchAnd) {
    milvus::QueryRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter(R"(TEXT_MATCH(text, "Euclidean") and TEXT_MATCH(text, "distance"))");
    req.AddOutputField("text");
    req.WithLimit(100);
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::QueryResponse resp;
    auto status = client_->Query(req, resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    resp.Results().OutputRows(rows);
    EXPECT_EQ(rows.size(), 1);
    auto text = rows[0]["text"].get<std::string>();
    EXPECT_NE(text.find("Euclidean"), std::string::npos);
    EXPECT_NE(text.find("distance"), std::string::npos);
}

TEST_F(MilvusServerTestTextMatch, SearchWithTextMatchFilter) {
    milvus::SearchRequest req;
    req.WithCollectionName(collection_name);
    req.WithAnnsField("vec");
    req.WithLimit(10);
    req.AddFloatVector({0.5f, 0.2f, 0.3f, 0.4f});
    req.WithFilter(R"(TEXT_MATCH(text, "vector database"))");
    req.AddOutputField("text");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResponse resp;
    auto status = client_->Search(req, resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_GE(results.at(0).Scores().size(), 1);

    milvus::EntityRows rows;
    results.at(0).OutputRows(rows);
    for (const auto& row : rows) {
        auto text = row["text"].get<std::string>();
        bool has_vector = text.find("vector") != std::string::npos;
        bool has_database = text.find("database") != std::string::npos;
        EXPECT_TRUE(has_vector || has_database) << "expected 'vector' or 'database' in: " << text;
    }
}

///////////////////////////////////////////////////////////////////////////////
// BM25 full text search tests: sparse vector generated by BM25 function
///////////////////////////////////////////////////////////////////////////////
class MilvusServerTestBM25 : public ::testing::Test {
 protected:
    static std::shared_ptr<milvus::MilvusClientV2> client_;
    static std::string collection_name;

    static void
    SetUpTestSuite() {
        const char* host = std::getenv("MILVUS_HOST");
        milvus::ConnectParam connect_param{host ? host : "localhost", 19530};
        client_ = milvus::MilvusClientV2::Create();
        auto status = client_->Connect(connect_param);
        milvus::test::ExpectStatusOK(status);

        collection_name = milvus::test::RanName("BM25_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, false));
        schema->AddField(milvus::FieldSchema("sparse_vec", milvus::DataType::SPARSE_FLOAT_VECTOR));
        schema->AddField(
            milvus::FieldSchema("text", milvus::DataType::VARCHAR, "text").WithMaxLength(65535).EnableAnalyzer(true));

        milvus::FunctionPtr bm25_func = std::make_shared<milvus::Function>("bm25_func", milvus::FunctionType::BM25);
        bm25_func->AddInputFieldName("text");
        bm25_func->AddOutputFieldName("sparse_vec");
        schema->AddFunction(bm25_func);

        status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        milvus::IndexDesc idx("sparse_vec", "", milvus::IndexType::SPARSE_INVERTED_INDEX, milvus::MetricType::BM25);
        status = client_->CreateIndex(
            milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(idx)));
        milvus::test::ExpectStatusOK(status);

        milvus::EntityRows rows_data;
        for (size_t i = 0; i < kTextContent.size(); ++i) {
            nlohmann::json row;
            row["id"] = static_cast<int64_t>(i);
            row["text"] = kTextContent[i];
            rows_data.emplace_back(std::move(row));
        }

        milvus::InsertRequest insert_req;
        insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
        milvus::InsertResponse insert_resp;
        status = client_->Insert(insert_req, insert_resp);
        milvus::test::ExpectStatusOK(status);

        status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
        milvus::test::ExpectStatusOK(status);
    }

    static void
    TearDownTestSuite() {
        if (client_) {
            client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
            client_->Disconnect();
            client_.reset();
        }
    }
};

std::shared_ptr<milvus::MilvusClientV2> MilvusServerTestBM25::client_;
std::string MilvusServerTestBM25::collection_name;

TEST_F(MilvusServerTestBM25, SearchByText) {
    milvus::SearchRequest req;
    req.WithCollectionName(collection_name);
    req.WithAnnsField("sparse_vec");
    req.WithLimit(5);
    req.AddEmbeddedText("Milvus vector database");
    req.AddOutputField("text");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResponse resp;
    auto status = client_->Search(req, resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_GE(results.at(0).Scores().size(), 1);

    milvus::EntityRows rows;
    results.at(0).OutputRows(rows);
    EXPECT_GT(rows.size(), 0);
    auto top_text = rows[0]["text"].get<std::string>();
    bool relevant = top_text.find("Milvus") != std::string::npos || top_text.find("vector") != std::string::npos ||
                    top_text.find("database") != std::string::npos;
    EXPECT_TRUE(relevant) << "top result not relevant: " << top_text;
}

TEST_F(MilvusServerTestBM25, SearchByMultipleTexts) {
    milvus::SearchRequest req;
    req.WithCollectionName(collection_name);
    req.WithAnnsField("sparse_vec");
    req.WithLimit(3);
    req.AddEmbeddedText("moon earth distance");
    req.AddEmbeddedText("electric car");
    req.AddOutputField("text");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResponse resp;
    auto status = client_->Search(req, resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = resp.Results().Results();
    EXPECT_EQ(results.size(), 2);

    milvus::EntityRows rows1;
    results.at(0).OutputRows(rows1);
    EXPECT_GT(rows1.size(), 0);
    auto text1 = rows1[0]["text"].get<std::string>();
    bool relevant1 = text1.find("moon") != std::string::npos || text1.find("earth") != std::string::npos ||
                     text1.find("distance") != std::string::npos;
    EXPECT_TRUE(relevant1) << "first query top result not relevant: " << text1;

    milvus::EntityRows rows2;
    results.at(1).OutputRows(rows2);
    EXPECT_GT(rows2.size(), 0);
    auto text2 = rows2[0]["text"].get<std::string>();
    bool relevant2 = text2.find("electric") != std::string::npos || text2.find("car") != std::string::npos;
    EXPECT_TRUE(relevant2) << "second query top result not relevant: " << text2;
}

///////////////////////////////////////////////////////////////////////////////
// Multi-analyzer tests: different analyzers for different languages
///////////////////////////////////////////////////////////////////////////////
class MilvusServerTestMultiAnalyzer : public ::testing::Test {
 protected:
    static std::shared_ptr<milvus::MilvusClientV2> client_;
    static std::string collection_name;

    static void
    SetUpTestSuite() {
        const char* host = std::getenv("MILVUS_HOST");
        milvus::ConnectParam connect_param{host ? host : "localhost", 19530};
        client_ = milvus::MilvusClientV2::Create();
        auto status = client_->Connect(connect_param);
        milvus::test::ExpectStatusOK(status);

        collection_name = milvus::test::RanName("MultiAna_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        schema->AddField(milvus::FieldSchema("sparse_vec", milvus::DataType::SPARSE_FLOAT_VECTOR));

        nlohmann::json multi_analyzers = {
            {"analyzers",
             {{"english", {{"type", "english"}}},
              {"chinese", {{"tokenizer", "jieba"}, {"filter", {"lowercase", "removepunct"}}}},
              {"japanese", {{"tokenizer", {{"type", "lindera"}, {"dict_kind", "ipadic"}}}}},
              {"default", {{"tokenizer", "icu"}, {"filter", {"lowercase", "removepunct", "asciifolding"}}}}}},
            {"by_field", "language"},
            {"alias", {{"cn", "chinese"}, {"en", "english"}, {"jp", "japanese"}}}};

        schema->AddField(milvus::FieldSchema("text", milvus::DataType::VARCHAR)
                             .WithMaxLength(65535)
                             .EnableAnalyzer(true)
                             .WithMultiAnalyzerParams(multi_analyzers));
        schema->AddField(milvus::FieldSchema("language", milvus::DataType::VARCHAR).WithMaxLength(64));

        milvus::FunctionPtr bm25_func = std::make_shared<milvus::Function>("bm25_func", milvus::FunctionType::BM25);
        bm25_func->AddInputFieldName("text");
        bm25_func->AddOutputFieldName("sparse_vec");
        schema->AddFunction(bm25_func);

        status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        milvus::IndexDesc idx("sparse_vec", "", milvus::IndexType::SPARSE_INVERTED_INDEX, milvus::MetricType::BM25);
        status = client_->CreateIndex(
            milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(idx)));
        milvus::test::ExpectStatusOK(status);

        milvus::EntityRows rows_data;
        for (const auto& text : std::vector<std::string>{
                 "Milvus is an open-source vector database", "AI applications help people better life",
                 "Newton is one of the greatest scientist of human history",
                 "Milvus supports L2 distance and IP similarity for float vector"}) {
            rows_data.push_back(nlohmann::json{{"text", text}, {"language", "en"}});
        }
        for (const auto& text : std::vector<std::string>{"人工智能正在改变技术领域", "机器学习模型需要大型数据集",
                                                         "Milvus 是一个高性能、可扩展的向量数据库"}) {
            rows_data.push_back(nlohmann::json{{"text", text}, {"language", "cn"}});
        }
        for (const auto& text : std::vector<std::string>{"Milvusの新機能をご確認ください",
                                                         "非構造化データやマルチモーダルデータを整理する"}) {
            rows_data.push_back(nlohmann::json{{"text", text}, {"language", "jp"}});
        }
        rows_data.push_back(nlohmann::json{{"text", "Les applications qui suivent le temps"}, {"language", "default"}});

        milvus::InsertRequest insert_req;
        insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
        milvus::InsertResponse insert_resp;
        status = client_->Insert(insert_req, insert_resp);
        milvus::test::ExpectStatusOK(status);

        status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
        milvus::test::ExpectStatusOK(status);
    }

    static void
    TearDownTestSuite() {
        if (client_) {
            client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
            client_->Disconnect();
            client_.reset();
        }
    }
};

std::shared_ptr<milvus::MilvusClientV2> MilvusServerTestMultiAnalyzer::client_;
std::string MilvusServerTestMultiAnalyzer::collection_name;

TEST_F(MilvusServerTestMultiAnalyzer, SearchEnglish) {
    milvus::SearchRequest req;
    req.WithCollectionName(collection_name);
    req.WithAnnsField("sparse_vec");
    req.WithLimit(5);
    req.AddEmbeddedText("Milvus vector database");
    req.AddExtraParam("analyzer_name", "english");
    req.AddOutputField("text");
    req.AddOutputField("language");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResponse resp;
    auto status = client_->Search(req, resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_GE(results.at(0).Scores().size(), 1);

    milvus::EntityRows rows;
    results.at(0).OutputRows(rows);
    auto top_text = rows[0]["text"].get<std::string>();
    bool relevant = top_text.find("Milvus") != std::string::npos || top_text.find("vector") != std::string::npos ||
                    top_text.find("database") != std::string::npos;
    EXPECT_TRUE(relevant) << "top result not relevant: " << top_text;
}

TEST_F(MilvusServerTestMultiAnalyzer, SearchChinese) {
    milvus::SearchRequest req;
    req.WithCollectionName(collection_name);
    req.WithAnnsField("sparse_vec");
    req.WithLimit(5);
    req.AddEmbeddedText("人工智能与机器学习");
    req.AddExtraParam("analyzer_name", "chinese");
    req.AddOutputField("text");
    req.AddOutputField("language");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResponse resp;
    auto status = client_->Search(req, resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_GE(results.at(0).Scores().size(), 1);

    milvus::EntityRows rows;
    results.at(0).OutputRows(rows);
    auto top_text = rows[0]["text"].get<std::string>();
    bool relevant = top_text.find("人工智能") != std::string::npos || top_text.find("机器学习") != std::string::npos;
    EXPECT_TRUE(relevant) << "top result not relevant: " << top_text;
}

TEST_F(MilvusServerTestMultiAnalyzer, SearchJapanese) {
    milvus::SearchRequest req;
    req.WithCollectionName(collection_name);
    req.WithAnnsField("sparse_vec");
    req.WithLimit(5);
    req.AddEmbeddedText("非構造化データ");
    req.AddExtraParam("analyzer_name", "japanese");
    req.AddOutputField("text");
    req.AddOutputField("language");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResponse resp;
    auto status = client_->Search(req, resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_GE(results.at(0).Scores().size(), 1);

    milvus::EntityRows rows;
    results.at(0).OutputRows(rows);
    auto top_text = rows[0]["text"].get<std::string>();
    bool relevant = top_text.find("データ") != std::string::npos;
    EXPECT_TRUE(relevant) << "top result not relevant: " << top_text;
}

TEST_F(MilvusServerTestMultiAnalyzer, SearchDefaultAnalyzer) {
    milvus::SearchRequest req;
    req.WithCollectionName(collection_name);
    req.WithAnnsField("sparse_vec");
    req.WithLimit(5);
    req.AddEmbeddedText("applications temps");
    req.AddExtraParam("analyzer_name", "default");
    req.AddOutputField("text");
    req.AddOutputField("language");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResponse resp;
    auto status = client_->Search(req, resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_GE(results.at(0).Scores().size(), 1);
}
