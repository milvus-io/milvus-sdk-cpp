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
    std::string collection_name;
    std::string partition_name;

    void
    createCollectionAndPartitions() {
        collection_name = milvus::test::RanName("Foo_");
        partition_name = milvus::test::RanName("Bar_");
        auto collection_schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        collection_schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        collection_schema->AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
        collection_schema->AddField(
            milvus::FieldSchema("face", milvus::DataType::BINARY_VECTOR, "face signature").WithDimension(32));

        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(
                collection_schema));
        EXPECT_EQ(status.Message(), "OK");
        milvus::test::ExpectStatusOK(status);

        status = client_->CreatePartition(
            milvus::CreatePartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name));
        EXPECT_EQ(status.Message(), "OK");
        milvus::test::ExpectStatusOK(status);
    }

    milvus::DmlResults
    insertRecords(const std::vector<milvus::FieldDataPtr>& fields) {
        milvus::InsertRequest insert_req;
        insert_req.WithCollectionName(collection_name).WithPartitionName(partition_name);
        std::vector<milvus::FieldDataPtr> fields_copy(fields);
        insert_req.WithColumnsData(std::move(fields_copy));

        milvus::InsertResponse insert_resp;
        auto status = client_->Insert(insert_req, insert_resp);
        EXPECT_EQ(status.Message(), "OK");
        milvus::test::ExpectStatusOK(status);

        const auto& dml_results = insert_resp.Results();
        EXPECT_EQ(dml_results.IdArray().IntIDArray().size(), fields.front()->Count());
        EXPECT_EQ(dml_results.InsertCount(), fields.front()->Count());
        return dml_results;
    }

    void
    loadCollection() {
        auto status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
        EXPECT_EQ(status.Message(), "OK");
        milvus::test::ExpectStatusOK(status);
    }

    void
    dropCollection() {
        auto status = client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
        milvus::test::ExpectStatusOK(status);
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
        faces.emplace_back(std::move(std::vector<uint8_t>{face_gen(rng), face_gen(rng), face_gen(rng), face_gen(rng)}));
    }

    std::vector<milvus::FieldDataPtr> fields{std::make_shared<milvus::Int16FieldData>("age", ages),
                                             std::make_shared<milvus::BinaryVecFieldData>("face", faces)};

    createCollectionAndPartitions();
    auto dml_results = insertRecords(fields);

    milvus::IndexDesc index_desc("face", "", milvus::IndexType::BIN_FLAT, milvus::MetricType::HAMMING);
    auto status = client_->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
    EXPECT_EQ(status.Message(), "OK");
    milvus::test::ExpectStatusOK(status);

    loadCollection();

    milvus::SearchRequest search_req{};
    search_req.WithCollectionName(collection_name);
    search_req.WithLimit(10);
    search_req.WithMetricType(milvus::MetricType::HAMMING);
    search_req.WithAnnsField("face");
    search_req.AddBinaryVector(std::vector<uint8_t>{255, 255, 255, 255});
    search_req.AddBinaryVector(std::vector<uint8_t>{0, 0, 0, 0});

    milvus::SearchResponse search_resp{};
    status = client_->Search(search_req, search_resp);
    EXPECT_EQ(status.Message(), "OK");
    milvus::test::ExpectStatusOK(status);

    const auto& results = search_resp.Results().Results();
    EXPECT_EQ(results.size(), 2);

    EXPECT_EQ(results.at(0).Scores().size(), 10);
    EXPECT_EQ(results.at(1).Scores().size(), 10);

    dropCollection();
}
