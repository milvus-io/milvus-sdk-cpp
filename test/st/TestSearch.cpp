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

class MilvusServerTestSearch : public MilvusServerTest {
 protected:
    std::string collection_name;
    std::string partition_name;

    void
    createCollectionAndPartitions(bool create_flat_index) {
        collection_name = milvus::test::RanName("Foo_");
        partition_name = milvus::test::RanName("Bar_");
        auto collection_schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        collection_schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        collection_schema->AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
        collection_schema->AddField(
            milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name").WithMaxLength(64));
        collection_schema->AddField(
            milvus::FieldSchema("face", milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(4));

        client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(
                collection_schema));
        EXPECT_EQ(status.Message(), "OK");
        milvus::test::ExpectStatusOK(status);

        if (create_flat_index) {
            milvus::IndexDesc index_desc("face", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
            status = client_->CreateIndex(
                milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
            EXPECT_EQ(status.Message(), "OK");
            milvus::test::ExpectStatusOK(status);
        }

        status = client_->CreatePartition(
            milvus::CreatePartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name));
        EXPECT_EQ(status.Message(), "OK");
        milvus::test::ExpectStatusOK(status);
    }

    milvus::DmlResults
    insertRecords(const std::vector<milvus::FieldDataPtr>& fields) {
        milvus::InsertRequest insert_req;
        insert_req.WithCollectionName(collection_name).WithPartitionName(partition_name);
        // make a copy since WithColumnsData takes rvalue
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

TEST_F(MilvusServerTestSearch, SearchWithoutIndex) {
    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::Int16FieldData>("age", std::vector<int16_t>{12, 13}),
        std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"Tom", "Jerry"}),
        std::make_shared<milvus::FloatVecFieldData>(
            "face", std::vector<std::vector<float>>{std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
                                                    std::vector<float>{0.5f, 0.6f, 0.7f, 0.8f}})};

    createCollectionAndPartitions(true);
    auto dml_results = insertRecords(fields);
    loadCollection();

    milvus::SearchRequest search_req{};
    search_req.WithCollectionName(collection_name);
    search_req.AddPartitionName(partition_name);
    search_req.WithLimit(10);
    search_req.AddOutputField("age");
    search_req.AddOutputField("name");
    search_req.WithFilter("id > 0");
    search_req.WithAnnsField("face");
    search_req.AddFloatVector(std::vector<float>{0.f, 0.f, 0.f, 0.f});
    search_req.AddFloatVector(std::vector<float>{1.f, 1.f, 1.f, 1.f});
    search_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::SearchResponse search_resp{};
    auto status = client_->Search(search_req, search_resp);
    EXPECT_EQ(status.Message(), "OK");
    milvus::test::ExpectStatusOK(status);

    const auto& results = search_resp.Results().Results();
    EXPECT_EQ(results.size(), 2);
    EXPECT_THAT(results.at(0).Ids().IntIDArray(), UnorderedElementsAreArray(dml_results.IdArray().IntIDArray()));
    EXPECT_THAT(results.at(1).Ids().IntIDArray(), UnorderedElementsAreArray(dml_results.IdArray().IntIDArray()));

    EXPECT_EQ(results.at(0).Scores().size(), 2);
    EXPECT_EQ(results.at(1).Scores().size(), 2);

    EXPECT_LT(results.at(0).Scores().at(0), results.at(0).Scores().at(1));
    EXPECT_LT(results.at(1).Scores().at(0), results.at(1).Scores().at(1));

    // match fields: id, score, age, name
    EXPECT_EQ(results.at(0).OutputFields().size(), 4);
    EXPECT_EQ(results.at(1).OutputFields().size(), 4);
    EXPECT_THAT(dynamic_cast<milvus::Int16FieldData&>(*results.at(0).OutputField("age")).Data(),
                UnorderedElementsAre(12, 13));
    EXPECT_THAT(dynamic_cast<milvus::Int16FieldData&>(*results.at(1).OutputField("age")).Data(),
                UnorderedElementsAre(12, 13));
    EXPECT_THAT(dynamic_cast<milvus::VarCharFieldData&>(*results.at(0).OutputField("name")).Data(),
                UnorderedElementsAre("Tom", "Jerry"));
    EXPECT_THAT(dynamic_cast<milvus::VarCharFieldData&>(*results.at(1).OutputField("name")).Data(),
                UnorderedElementsAre("Tom", "Jerry"));
    dropCollection();
}

TEST_F(MilvusServerTestSearch, RangeSearch) {
    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::Int16FieldData>("age", std::vector<int16_t>{12, 13, 14, 15, 16, 17, 18}),
        std::make_shared<milvus::VarCharFieldData>(
            "name", std::vector<std::string>{"Tom", "Jerry", "Lily", "Foo", "Bar", "Jake", "Jonathon"}),
        std::make_shared<milvus::FloatVecFieldData>("face", std::vector<std::vector<float>>{
                                                                std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
                                                                std::vector<float>{0.2f, 0.3f, 0.4f, 0.5f},
                                                                std::vector<float>{0.3f, 0.4f, 0.5f, 0.6f},
                                                                std::vector<float>{0.4f, 0.5f, 0.6f, 0.7f},
                                                                std::vector<float>{0.5f, 0.6f, 0.7f, 0.8f},
                                                                std::vector<float>{0.6f, 0.7f, 0.8f, 0.9f},
                                                                std::vector<float>{0.7f, 0.8f, 0.9f, 1.0f},
                                                            })};

    createCollectionAndPartitions(true);
    auto dml_results = insertRecords(fields);
    loadCollection();

    milvus::SearchRequest search_req{};
    search_req.WithCollectionName(collection_name);
    search_req.AddPartitionName(partition_name);
    search_req.WithRangeFilter(0.3);
    search_req.WithRadius(1.0);
    search_req.WithLimit(10);
    search_req.AddOutputField("age");
    search_req.AddOutputField("name");
    search_req.WithAnnsField("face");
    search_req.AddFloatVector(std::vector<float>{0.f, 0.f, 0.f, 0.f});
    search_req.AddFloatVector(std::vector<float>{1.f, 1.f, 1.f, 1.f});
    search_req.WithConsistencyLevel(milvus::ConsistencyLevel::SESSION);

    milvus::SearchResponse search_resp{};
    auto status = client_->Search(search_req, search_resp);
    EXPECT_EQ(status.Message(), "OK");
    milvus::test::ExpectStatusOK(status);

    const auto& results = search_resp.Results().Results();
    EXPECT_EQ(results.size(), 2);

    // validate results
    auto validateScores = [&results](int firstRet, int secondRet) {
        // check score should between range
        for (const auto& result : results) {
            for (const auto& score : result.Scores()) {
                EXPECT_GE(score, 0.3);
                EXPECT_LE(score, 1.0);
            }
        }
        EXPECT_EQ(results.at(0).Ids().IntIDArray().size(), firstRet);
        EXPECT_EQ(results.at(1).Ids().IntIDArray().size(), secondRet);
    };

    // valid score in range is 3, 2
    validateScores(3, 2);

    // add fields, then search again, should be 6 and 4
    insertRecords(fields);
    loadCollection();
    status = client_->Search(search_req, search_resp);
    milvus::test::ExpectStatusOK(status);
    validateScores(6, 4);

    // add fields twice, and now it should be 12, 8, as limit is 10, then should be 10, 8
    insertRecords(fields);
    insertRecords(fields);
    loadCollection();
    status = client_->Search(search_req, search_resp);
    milvus::test::ExpectStatusOK(status);
    validateScores(10, 8);

    dropCollection();
}

TEST_F(MilvusServerTestSearch, SearchWithStringFilter) {
    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::Int16FieldData>("age", std::vector<int16_t>{12, 13}),
        std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"Tom", "Jerry"}),
        std::make_shared<milvus::FloatVecFieldData>(
            "face", std::vector<std::vector<float>>{std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
                                                    std::vector<float>{0.5f, 0.6f, 0.7f, 0.8f}})};

    createCollectionAndPartitions(true);
    auto dml_results = insertRecords(fields);
    loadCollection();

    milvus::SearchRequest search_req{};
    search_req.WithCollectionName(collection_name);
    search_req.AddPartitionName(partition_name);
    search_req.WithLimit(10);
    search_req.AddOutputField("age");
    search_req.AddOutputField("name");
    search_req.WithFilter("name like \"To%\"");  // Tom match To%
    search_req.WithAnnsField("face");
    search_req.AddFloatVector(std::vector<float>{0.f, 0.f, 0.f, 0.f});
    search_req.AddFloatVector(std::vector<float>{1.f, 1.f, 1.f, 1.f});
    search_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::SearchResponse search_resp{};
    auto status = client_->Search(search_req, search_resp);
    EXPECT_EQ(status.Message(), "OK");
    milvus::test::ExpectStatusOK(status);

    const auto& results = search_resp.Results().Results();
    EXPECT_EQ(results.size(), 2);

    EXPECT_EQ(results.at(0).Scores().size(), 1);
    EXPECT_EQ(results.at(1).Scores().size(), 1);

    // match fields: id, score, age, name
    EXPECT_EQ(results.at(0).OutputFields().size(), 4);
    EXPECT_EQ(results.at(1).OutputFields().size(), 4);
    EXPECT_EQ(dynamic_cast<milvus::Int16FieldData&>(*results.at(0).OutputField("age")).Data(),
              std::vector<int16_t>{12});
    EXPECT_EQ(dynamic_cast<milvus::Int16FieldData&>(*results.at(1).OutputField("age")).Data(),
              std::vector<int16_t>{12});
    EXPECT_EQ(dynamic_cast<milvus::VarCharFieldData&>(*results.at(0).OutputField("name")).Data(),
              std::vector<std::string>{"Tom"});
    EXPECT_EQ(dynamic_cast<milvus::VarCharFieldData&>(*results.at(1).OutputField("name")).Data(),
              std::vector<std::string>{"Tom"});
    dropCollection();
}

// for issue #158
TEST_F(MilvusServerTestSearch, SearchWithIVFIndex) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int16_t> age_gen{10, 30};
    std::uniform_real_distribution<float> face_gen{0.f, 1.f};
    size_t test_count = 1000;
    std::vector<int16_t> ages{};
    std::vector<std::vector<float>> faces{};
    std::vector<std::string> names{};
    for (auto i = test_count; i > 0; --i) {
        ages.emplace_back(age_gen(rng));
        names.emplace_back(std::move(std::string("name_") + std::to_string(i)));
        faces.emplace_back(std::move(std::vector<float>{face_gen(rng), face_gen(rng), face_gen(rng), face_gen(rng)}));
    }

    std::vector<milvus::FieldDataPtr> fields{std::make_shared<milvus::Int16FieldData>("age", ages),
                                             std::make_shared<milvus::VarCharFieldData>("name", names),
                                             std::make_shared<milvus::FloatVecFieldData>("face", faces)};

    createCollectionAndPartitions(false);
    auto dml_results = insertRecords(fields);

    milvus::IndexDesc index_desc("face", "", milvus::IndexType::IVF_FLAT, milvus::MetricType::L2);
    index_desc.AddExtraParam("nlist", "1024");
    auto status = client_->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
    EXPECT_EQ(status.Message(), "OK");
    milvus::test::ExpectStatusOK(status);

    loadCollection();

    milvus::SearchRequest search_req{};
    search_req.WithCollectionName(collection_name);
    search_req.WithLimit(10);
    search_req.WithMetricType(milvus::MetricType::L2);
    search_req.AddExtraParam("nprobe", "10");
    search_req.WithAnnsField("face");
    search_req.AddFloatVector(std::vector<float>{0.f, 0.f, 0.f, 0.f});
    search_req.AddFloatVector(std::vector<float>{1.f, 1.f, 1.f, 1.f});
    search_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

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
