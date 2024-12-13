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

using milvus::test::MilvusServerTest;
using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

class MilvusServerTestSearchWithVectors : public MilvusServerTest {
 protected:
    std::string collection_name{"Foo"};
    std::string partition_name{"Bar"};
    static constexpr int DIMENSION = 32;

    void
    createCollectionAndPartitions() {
        milvus::CollectionSchema collection_schema(collection_name);
        collection_schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        collection_schema.AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
        collection_schema.AddField(
            milvus::FieldSchema("face", vector_data_type, "face signature").WithDimension(DIMENSION));

        auto status = client_->CreateCollection(collection_schema);
        EXPECT_EQ(status.Message(), "OK");
        EXPECT_TRUE(status.IsOk());

        status = client_->CreatePartition(collection_name, partition_name);
        EXPECT_EQ(status.Message(), "OK");
        EXPECT_TRUE(status.IsOk());
    }

    milvus::DmlResults
    insertRecords(const std::vector<milvus::FieldDataPtr>& fields) {
        milvus::DmlResults dml_results;
        auto status = client_->Insert(collection_name, partition_name, fields, dml_results);
        EXPECT_EQ(status.Message(), "OK");
        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(dml_results.IdArray().IntIDArray().size(), fields.front()->Count());
        return dml_results;
    }

    void
    loadCollection() {
        auto status = client_->LoadCollection(collection_name);
        EXPECT_EQ(status.Message(), "OK");
        EXPECT_TRUE(status.IsOk());
    }

    void
    dropCollection() {
        auto status = client_->DropCollection(collection_name);
        EXPECT_TRUE(status.IsOk());
    }
};

// for issue #194
TEST_F(MilvusServerTestSearchWithVectors, RegressionIssue194) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int16_t> age_gen{10, 30};
    std::uniform_int_distribution<uint8_t> face_gen{0, 255};
    size_t test_count = 1000;
    std::vector<int16_t> ages{};
    std::vector<std::vector<uint8_t>> faces{};
    for (auto i = test_count; i > 0; --i) {
        ages.emplace_back(age_gen(rng));
        faces.emplace_back(std::vector<uint8_t>{face_gen(rng), face_gen(rng), face_gen(rng), face_gen(rng)});
    }

    std::vector<milvus::FieldDataPtr> fields{std::make_shared<milvus::Int16FieldData>("age", ages),
                                             std::make_shared<milvus::BinaryVecFieldData>("face", faces)};

    createCollectionAndPartitions(milvus::DataType::BINARY_VECTOR);
    auto dml_results = insertRecords(fields);

    milvus::IndexDesc index_desc("face", "", milvus::IndexType::BIN_FLAT, milvus::MetricType::HAMMING, 0);
    auto status = client_->CreateIndex(collection_name, index_desc);
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_TRUE(status.IsOk());

    loadCollection();

    milvus::SearchArguments arguments{};
    arguments.SetCollectionName(collection_name);
    arguments.SetTopK(10);
    arguments.SetMetricType(milvus::MetricType::HAMMING);
    arguments.AddTargetVector<milvus::BinaryVecFieldData>("face", std::vector<uint8_t>{255, 255, 255, 255});
    arguments.AddTargetVector<milvus::BinaryVecFieldData>("face", std::vector<uint8_t>{0, 0, 0, 0});
    milvus::SearchResults search_results{};
    status = client_->Search(arguments, search_results);
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_TRUE(status.IsOk());

    const auto& results = search_results.Results();
    EXPECT_EQ(results.size(), 2);

    EXPECT_EQ(results.at(0).Scores().size(), 10);
    EXPECT_EQ(results.at(1).Scores().size(), 10);

    dropCollection();
}

// milvus lite does not support bfloat16 vector
TEST_F(MilvusServerTestSearchWithVectors, Float16Vector) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int16_t> age_gen{10, 30};
    std::uniform_real_distribution<double> face_gen{0, 255};
    size_t test_count = 10;
    std::vector<int16_t> ages{};
    std::vector<std::vector<double>> faces{};
    for (auto i = test_count; i > 0; --i) {
        ages.emplace_back(age_gen(rng));
        faces.emplace_back(std::vector<double>(DIMENSION, face_gen(rng)));
    }
    auto faces_field = std::make_shared<milvus::Float16VecFieldData>("face", faces);
    ASSERT_EQ(faces_field->DataAsFloats<float>()[0].size(), DIMENSION);
    std::vector<milvus::FieldDataPtr> fields{std::make_shared<milvus::Int16FieldData>("age", ages), faces_field};

    createCollectionAndPartitions(milvus::DataType::FLOAT16_VECTOR);
    auto dml_results = insertRecords(fields);

    milvus::IndexDesc index_desc("face", "", milvus::IndexType::FLAT, milvus::MetricType::L2, 0);
    auto status = client_->CreateIndex(collection_name, index_desc);
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_TRUE(status.IsOk());

    loadCollection();

    milvus::SearchArguments arguments{};
    arguments.SetCollectionName(collection_name);
    arguments.SetTopK(10);
    arguments.SetMetricType(milvus::MetricType::L2);
    arguments.AddTargetVector<milvus::Float16VecFieldData>("face", std::vector<double>(DIMENSION, 255));
    arguments.AddTargetVector<milvus::Float16VecFieldData>("face", std::vector<double>(DIMENSION, 0));
    milvus::SearchResults search_results{};
    status = client_->Search(arguments, search_results);
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_TRUE(status.IsOk());

    const auto& results = search_results.Results();
    EXPECT_EQ(results.size(), 2);

    EXPECT_EQ(results.at(0).Scores().size(), 10);
    EXPECT_EQ(results.at(1).Scores().size(), 10);

    dropCollection();
}
