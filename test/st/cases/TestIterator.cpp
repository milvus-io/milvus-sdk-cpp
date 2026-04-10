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

#include "MilvusServerTest.h"

class MilvusServerTestIterator : public ::testing::Test {
 protected:
    static std::shared_ptr<milvus::MilvusClientV2> client_;
    static std::string collection_name;
    static constexpr size_t total_count = 500;

    static void
    SetUpTestSuite() {
        const char* host = std::getenv("MILVUS_HOST");
        milvus::ConnectParam connect_param{host ? host : "localhost", 19530};
        client_ = milvus::MilvusClientV2::Create();
        auto status = client_->Connect(connect_param);
        milvus::test::ExpectStatusOK(status);

        collection_name = milvus::test::RanName("IterTest_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        schema->AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
        schema->AddField(milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name").WithMaxLength(64));
        schema->AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(4));

        status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        milvus::IndexDesc index_desc("vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
        status = client_->CreateIndex(
            milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
        milvus::test::ExpectStatusOK(status);

        // insert data
        std::mt19937 rng(42);
        std::uniform_int_distribution<int16_t> age_gen{10, 60};
        std::uniform_real_distribution<float> vec_gen{0.f, 1.f};

        std::vector<int16_t> ages;
        std::vector<std::string> names;
        std::vector<std::vector<float>> vecs;
        for (size_t i = 0; i < total_count; ++i) {
            ages.push_back(age_gen(rng));
            names.push_back("name_" + std::to_string(i));
            vecs.push_back({vec_gen(rng), vec_gen(rng), vec_gen(rng), vec_gen(rng)});
        }

        std::vector<milvus::FieldDataPtr> fields{std::make_shared<milvus::Int16FieldData>("age", ages),
                                                 std::make_shared<milvus::VarCharFieldData>("name", names),
                                                 std::make_shared<milvus::FloatVecFieldData>("vec", vecs)};

        milvus::InsertRequest insert_req;
        insert_req.WithCollectionName(collection_name).WithColumnsData(std::move(fields));
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

std::shared_ptr<milvus::MilvusClientV2> MilvusServerTestIterator::client_;
std::string MilvusServerTestIterator::collection_name;
constexpr size_t MilvusServerTestIterator::total_count;

TEST_F(MilvusServerTestIterator, QueryIterator) {
    auto test_func = [&](int64_t batch_size, int64_t limit) {
        milvus::QueryIteratorRequest req;
        req.WithCollectionName(collection_name);
        req.WithFilter("age >= 0");
        req.AddOutputField("age");
        req.AddOutputField("name");
        req.SetBatchSize(batch_size);
        req.SetLimit(limit);
        req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        milvus::QueryIteratorPtr iterator;
        auto status = client_->QueryIterator(req, iterator);
        milvus::test::ExpectStatusOK(status);
        ASSERT_NE(iterator, nullptr);

        // iterate through all results
        size_t total_fetched = 0;
        while (true) {
            milvus::QueryResults batch;
            status = iterator->Next(batch);
            milvus::test::ExpectStatusOK(status);

            auto batch_count = batch.GetRowCount();
            if (batch_count == 0) {
                std::cout << "query iteration finished" << std::endl;
                break;
            }

            auto row_count = batch.OutputFields().front()->Count();
            EXPECT_GT(row_count, 0);
            EXPECT_EQ(row_count, batch_count);
            total_fetched += row_count;

            milvus::EntityRows rows;
            status = batch.OutputRows(rows);
            milvus::test::ExpectStatusOK(status);

            for (const auto& row : rows) {
                EXPECT_TRUE(row.contains("age"));
                EXPECT_TRUE(row.contains("name"));

                auto age = row["age"].get<int16_t>();
                auto name = row["name"].get<std::string>();
                EXPECT_TRUE(age >= 10 && age <= 60);
                EXPECT_TRUE(name.rfind("name_", 0) == 0);  // name starts with "name_"
            }
        }

        if (limit < 0) {
            EXPECT_EQ(total_fetched, total_count);
        } else if (limit == 0) {
            EXPECT_EQ(total_fetched, 0);
        } else if (limit < total_count) {
            EXPECT_EQ(total_fetched, limit);
        } else {
            EXPECT_EQ(total_fetched, total_count);
        }
        return;
    };

    test_func(100, -1);
    test_func(100, 0);
    test_func(1000, -1);
    test_func(1000, 5);
}

TEST_F(MilvusServerTestIterator, QueryIteratorWithFilter) {
    milvus::QueryIteratorRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter("age > 30");
    req.AddOutputField("age");
    req.SetBatchSize(50);
    // req.SetLimit(-1);
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::QueryIteratorPtr iterator;
    auto status = client_->QueryIterator(req, iterator);
    milvus::test::ExpectStatusOK(status);
    ASSERT_NE(iterator, nullptr);

    size_t total_fetched = 0;
    while (true) {
        milvus::QueryResults batch;
        status = iterator->Next(batch);
        milvus::test::ExpectStatusOK(status);

        auto batch_count = batch.GetRowCount();
        if (batch_count == 0) {
            std::cout << "query iteration finished" << std::endl;
            break;
        }

        // verify all returned ages are > 30
        auto age_field = std::dynamic_pointer_cast<milvus::Int16FieldData>(batch.OutputField("age"));
        ASSERT_NE(age_field, nullptr);
        for (const auto& age : age_field->Data()) {
            EXPECT_GT(age, 30);
        }

        total_fetched += age_field->Count();
    }

    EXPECT_GT(total_fetched, 0);
    EXPECT_LT(total_fetched, total_count);
}

TEST_F(MilvusServerTestIterator, SearchIterator) {
    auto test_func = [&](int64_t batch_size, int64_t limit) {
        milvus::SearchIteratorRequest req;
        req.WithCollectionName(collection_name);
        req.WithAnnsField("vec");
        req.AddFloatVector({0.5f, 0.5f, 0.5f, 0.5f});
        req.SetBatchSize(batch_size);
        req.AddOutputField("age");
        req.WithLimit(limit);
        req.WithMetricType(milvus::MetricType::L2);
        req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        milvus::SearchIteratorPtr iterator;
        auto status = client_->SearchIterator(req, iterator);
        milvus::test::ExpectStatusOK(status);
        ASSERT_NE(iterator, nullptr);

        // iterate through results
        size_t total_fetched = 0;
        float prev_max_score = -1.f;
        while (true) {
            milvus::SingleResult batch;
            status = iterator->Next(batch);
            milvus::test::ExpectStatusOK(status);

            auto batch_count = batch.GetRowCount();
            if (batch_count == 0) {
                std::cout << "search iteration finished" << std::endl;
                break;
            }

            // scores should be non-decreasing across batches (L2 distance)
            EXPECT_GE(batch.Scores().front(), prev_max_score);
            prev_max_score = batch.Scores().back();

            auto row_count = batch.OutputFields().front()->Count();
            EXPECT_GT(row_count, 0);
            EXPECT_EQ(row_count, batch_count);
            total_fetched += row_count;

            milvus::EntityRows rows;
            status = batch.OutputRows(rows);
            milvus::test::ExpectStatusOK(status);

            for (const auto& row : rows) {
                EXPECT_TRUE(row.contains("age"));
                auto age = row["age"].get<int16_t>();
                EXPECT_TRUE(age >= 10 && age <= 60);
            }
        }

        if (limit < 0) {
            EXPECT_EQ(total_fetched, total_count);
        } else if (limit == 0) {
            EXPECT_EQ(total_fetched, 0);
        } else if (limit < total_count) {
            EXPECT_EQ(total_fetched, limit);
        } else {
            EXPECT_EQ(total_fetched, total_count);
        }
    };

    test_func(100, -1);
    test_func(100, 0);
    test_func(1000, -1);
    test_func(1000, 5);
}

TEST_F(MilvusServerTestIterator, SearchIteratorWithFilter) {
    milvus::SearchIteratorRequest req;
    req.WithCollectionName(collection_name);
    req.WithAnnsField("vec");
    req.AddFloatVector({0.5f, 0.5f, 0.5f, 0.5f});
    // req.WithLimit(total_count);
    req.SetBatchSize(50);
    req.WithMetricType(milvus::MetricType::L2);
    req.WithFilter("age <= 30");
    req.AddOutputField("age");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchIteratorPtr iterator;
    auto status = client_->SearchIterator(req, iterator);
    milvus::test::ExpectStatusOK(status);
    ASSERT_NE(iterator, nullptr);

    size_t total_fetched = 0;
    while (true) {
        milvus::SingleResult batch;
        status = iterator->Next(batch);
        milvus::test::ExpectStatusOK(status);

        if (batch.GetRowCount() == 0) {
            break;
        }

        total_fetched += batch.GetRowCount();
    }

    EXPECT_GT(total_fetched, 0);
    EXPECT_LT(total_fetched, total_count);
}
