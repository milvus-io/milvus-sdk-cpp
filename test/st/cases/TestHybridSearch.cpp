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

#include <random>
#include <set>

#include "MilvusServerTest.h"

class MilvusServerTestHybridSearch : public ::testing::Test {
 protected:
    static std::shared_ptr<milvus::MilvusClientV2> client_;
    static std::string collection_name;
    static constexpr uint32_t dimension = 8;
    static constexpr int row_count = 100;

    static void
    SetUpTestSuite() {
        const char* host = std::getenv("MILVUS_HOST");
        milvus::ConnectParam connect_param{host ? host : "localhost", 19530};
        client_ = milvus::MilvusClientV2::Create();
        auto status = client_->Connect(connect_param);
        milvus::test::ExpectStatusOK(status);

        collection_name = milvus::test::RanName("HybridTest_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, false));
        schema->AddField(milvus::FieldSchema("category", milvus::DataType::INT32, "category"));
        schema->AddField(milvus::FieldSchema("label", milvus::DataType::VARCHAR, "label").WithMaxLength(64));
        schema->AddField(
            milvus::FieldSchema("dense", milvus::DataType::FLOAT_VECTOR, "dense vector").WithDimension(dimension));
        schema->AddField(milvus::FieldSchema("sparse", milvus::DataType::SPARSE_FLOAT_VECTOR, "sparse vector"));

        status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        milvus::IndexDesc idx_dense("dense", "", milvus::IndexType::FLAT, milvus::MetricType::COSINE);
        milvus::IndexDesc idx_sparse("sparse", "", milvus::IndexType::SPARSE_INVERTED_INDEX, milvus::MetricType::IP);
        idx_sparse.AddExtraParam("drop_ratio_build", "0.2");
        status = client_->CreateIndex(milvus::CreateIndexRequest()
                                          .WithCollectionName(collection_name)
                                          .AddIndex(std::move(idx_dense))
                                          .AddIndex(std::move(idx_sparse)));
        milvus::test::ExpectStatusOK(status);

        // insert data
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> float_gen{0.f, 1.f};
        milvus::EntityRows rows_data;
        for (int i = 0; i < row_count; ++i) {
            nlohmann::json row;
            row["id"] = i;
            row["category"] = i % 5;
            row["label"] = "label_" + std::to_string(i);

            std::vector<float> dense_vec(dimension);
            for (auto& v : dense_vec) v = float_gen(rng);
            row["dense"] = dense_vec;

            nlohmann::json sparse;
            sparse[std::to_string(i % 50)] = float_gen(rng);
            sparse[std::to_string(i % 50 + 50)] = float_gen(rng);
            row["sparse"] = sparse;

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

    static std::vector<float>
    queryVector() {
        return std::vector<float>(dimension, 0.5f);
    }

    static std::map<uint32_t, float>
    querySparseVector() {
        return {{10, 0.8f}, {60, 0.5f}};
    }
};

std::shared_ptr<milvus::MilvusClientV2> MilvusServerTestHybridSearch::client_;
std::string MilvusServerTestHybridSearch::collection_name;

TEST_F(MilvusServerTestHybridSearch, WithRRFRerank) {
    auto sub1 = std::make_shared<milvus::SubSearchRequest>();
    sub1->WithAnnsField("dense").WithLimit(10).AddFloatVector(queryVector());

    auto sub2 = std::make_shared<milvus::SubSearchRequest>();
    sub2->WithAnnsField("sparse").WithLimit(10).WithMetricType(milvus::MetricType::IP);
    sub2->AddSparseVector(querySparseVector());

    milvus::HybridSearchRequest req;
    req.WithCollectionName(collection_name);
    req.AddSubRequest(sub1);
    req.AddSubRequest(sub2);
    req.WithRerank(std::make_shared<milvus::RRFRerank>(60));
    req.WithLimit(10);
    req.AddOutputField("category");
    req.AddOutputField("label");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::HybridSearchResponse resp;
    auto status = client_->HybridSearch(req, resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results.at(0).Scores().size(), 10);

    // verify output fields are present and valid
    milvus::EntityRows rows;
    results.at(0).OutputRows(rows);
    EXPECT_EQ(rows.size(), 10);
    for (const auto& row : rows) {
        EXPECT_TRUE(row.contains("category"));
        EXPECT_TRUE(row.contains("label"));
        auto cat = row["category"].get<int32_t>();
        EXPECT_GE(cat, 0);
        EXPECT_LE(cat, 4);
        auto label = row["label"].get<std::string>();
        EXPECT_TRUE(label.rfind("label_", 0) == 0);
    }
}

TEST_F(MilvusServerTestHybridSearch, WithWeightedRerank) {
    auto sub1 = std::make_shared<milvus::SubSearchRequest>();
    sub1->WithAnnsField("dense").WithLimit(10).AddFloatVector(queryVector());

    auto sub2 = std::make_shared<milvus::SubSearchRequest>();
    sub2->WithAnnsField("sparse").WithLimit(10).WithMetricType(milvus::MetricType::IP);
    sub2->AddSparseVector(querySparseVector());

    milvus::HybridSearchRequest req;
    req.WithCollectionName(collection_name);
    req.AddSubRequest(sub1);
    req.AddSubRequest(sub2);
    req.WithRerank(std::make_shared<milvus::WeightedRerank>(std::vector<float>{0.7f, 0.3f}));
    req.WithLimit(5);
    req.AddOutputField("label");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::HybridSearchResponse resp;
    auto status = client_->HybridSearch(req, resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results.at(0).Scores().size(), 5);
}

TEST_F(MilvusServerTestHybridSearch, SubRequestsWithFilters) {
    auto sub1 = std::make_shared<milvus::SubSearchRequest>();
    sub1->WithAnnsField("dense").WithLimit(10).WithFilter("category == 0");
    sub1->AddFloatVector(queryVector());

    auto sub2 = std::make_shared<milvus::SubSearchRequest>();
    sub2->WithAnnsField("sparse").WithLimit(10).WithMetricType(milvus::MetricType::IP);
    sub2->WithFilter("category in [1, 2]");
    sub2->AddSparseVector(querySparseVector());

    milvus::HybridSearchRequest req;
    req.WithCollectionName(collection_name);
    req.AddSubRequest(sub1);
    req.AddSubRequest(sub2);
    req.WithRerank(std::make_shared<milvus::RRFRerank>());
    req.WithLimit(10);
    req.AddOutputField("category");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::HybridSearchResponse resp;
    auto status = client_->HybridSearch(req, resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_GT(results.at(0).Scores().size(), 0);

    // verify all results have category 0, 1, or 2
    milvus::EntityRows rows;
    results.at(0).OutputRows(rows);
    for (const auto& row : rows) {
        auto cat = row["category"].get<int32_t>();
        EXPECT_TRUE(cat == 0 || cat == 1 || cat == 2) << "unexpected category: " << cat;
    }
}

TEST_F(MilvusServerTestHybridSearch, WithOffset) {
    auto sub1 = std::make_shared<milvus::SubSearchRequest>();
    sub1->WithAnnsField("dense").WithLimit(20).AddFloatVector(queryVector());

    auto sub2 = std::make_shared<milvus::SubSearchRequest>();
    sub2->WithAnnsField("sparse").WithLimit(20).WithMetricType(milvus::MetricType::IP);
    sub2->AddSparseVector(querySparseVector());

    // first page
    milvus::HybridSearchRequest req1;
    req1.WithCollectionName(collection_name);
    req1.AddSubRequest(sub1);
    req1.AddSubRequest(sub2);
    req1.WithRerank(std::make_shared<milvus::RRFRerank>(60));
    req1.WithLimit(5);
    req1.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::HybridSearchResponse resp1;
    auto status = client_->HybridSearch(req1, resp1);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(resp1.Results().Results().at(0).Scores().size(), 5);

    // second page with offset
    milvus::HybridSearchRequest req2;
    req2.WithCollectionName(collection_name);
    req2.AddSubRequest(sub1);
    req2.AddSubRequest(sub2);
    req2.WithRerank(std::make_shared<milvus::RRFRerank>(60));
    req2.WithLimit(5);
    req2.WithOffset(5);
    req2.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::HybridSearchResponse resp2;
    status = client_->HybridSearch(req2, resp2);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(resp2.Results().Results().at(0).Scores().size(), 5);

    // verify two pages return different IDs
    auto ids1 = resp1.Results().Results().at(0).Ids().IntIDArray();
    auto ids2 = resp2.Results().Results().at(0).Ids().IntIDArray();
    for (auto id : ids1) {
        EXPECT_TRUE(std::find(ids2.begin(), ids2.end(), id) == ids2.end());
    }
}

TEST_F(MilvusServerTestHybridSearch, WithGroupBy) {
    auto sub1 = std::make_shared<milvus::SubSearchRequest>();
    sub1->WithAnnsField("dense").WithLimit(20).AddFloatVector(queryVector());

    auto sub2 = std::make_shared<milvus::SubSearchRequest>();
    sub2->WithAnnsField("sparse").WithLimit(20).WithMetricType(milvus::MetricType::IP);
    sub2->AddSparseVector(querySparseVector());

    milvus::HybridSearchRequest req;
    req.WithCollectionName(collection_name);
    req.AddSubRequest(sub1);
    req.AddSubRequest(sub2);
    req.WithRerank(std::make_shared<milvus::RRFRerank>(60));
    req.WithLimit(10);
    req.WithGroupByField("category");
    req.AddOutputField("category");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::HybridSearchResponse resp;
    auto status = client_->HybridSearch(req, resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = resp.Results().Results();
    EXPECT_EQ(results.size(), 1);

    // with group_by on category (5 distinct values), each group gets 1 result by default
    milvus::EntityRows rows;
    results.at(0).OutputRows(rows);
    std::set<int32_t> seen_categories;
    for (const auto& row : rows) {
        seen_categories.insert(row["category"].get<int32_t>());
    }
    // all 5 categories should be represented
    EXPECT_EQ(seen_categories.size(), 5);
    // each category appears once (default group_size=1), so total results == number of categories
    EXPECT_EQ(rows.size(), 5);
}
