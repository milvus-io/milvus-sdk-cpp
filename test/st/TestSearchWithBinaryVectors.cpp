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

class MilvusServerTestSearchWithBinaryVectors : public MilvusServerTest {
 protected:
    std::string collection_name{"Foo"};
    std::string partition_name{"Bar"};

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
        client_->Connect(connect_param);
    }

    void
    TearDown() override {
        MilvusServerTest::TearDown();
    }

    void
    createCollectionAndPartitions() {
        milvus::CollectionSchema collection_schema(collection_name);
        collection_schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        collection_schema.AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
        collection_schema.AddField(
            milvus::FieldSchema("face", milvus::DataType::BINARY_VECTOR, "face signature").WithDimension(32));

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
TEST_F(MilvusServerTestSearchWithBinaryVectors, RegressionIssue194) {
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

    createCollectionAndPartitions();
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
    arguments.AddTargetVector("face", std::vector<uint8_t>{255, 255, 255, 255});
    arguments.AddTargetVector("face", std::vector<uint8_t>{0, 0, 0, 0});
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
