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
#include "milvus/utils/FP16.h"
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
        collection_schema->AddField(milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name").WithMaxLength(64));
        collection_schema->AddField(
            milvus::FieldSchema("face", milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(4));

        client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
        auto status = client_->CreateCollection(milvus::CreateCollectionRequest()
                                                    .WithCollectionName(collection_name)
                                                    .WithCollectionSchema(collection_schema));
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

TEST_F(MilvusServerTestSearch, SearchWithMultipleVectorTypes) {
    // create a collection with 4 vector fields of different types
    std::string coll_name = milvus::test::RanName("MultiVec_");
    auto schema = std::make_shared<milvus::CollectionSchema>(coll_name);
    schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
    schema->AddField(milvus::FieldSchema("float_vec", milvus::DataType::FLOAT_VECTOR, "float vector").WithDimension(4));
    schema->AddField(
        milvus::FieldSchema("binary_vec", milvus::DataType::BINARY_VECTOR, "binary vector").WithDimension(32));
    schema->AddField(
        milvus::FieldSchema("fp16_vec", milvus::DataType::FLOAT16_VECTOR, "float16 vector").WithDimension(4));
    schema->AddField(milvus::FieldSchema("sparse_vec", milvus::DataType::SPARSE_FLOAT_VECTOR, "sparse vector"));

    client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(coll_name));
    auto status = client_->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(coll_name).WithCollectionSchema(schema));
    milvus::test::ExpectStatusOK(status);

    // create indexes for each vector field
    milvus::IndexDesc float_idx("float_vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    milvus::IndexDesc binary_idx("binary_vec", "", milvus::IndexType::BIN_FLAT, milvus::MetricType::HAMMING);
    milvus::IndexDesc fp16_idx("fp16_vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    milvus::IndexDesc sparse_idx("sparse_vec", "", milvus::IndexType::SPARSE_INVERTED_INDEX, milvus::MetricType::IP);
    sparse_idx.AddExtraParam("drop_ratio_build", "0.2");

    status =
        client_->CreateIndex(milvus::CreateIndexRequest().WithCollectionName(coll_name).AddIndex(std::move(float_idx)));
    milvus::test::ExpectStatusOK(status);
    status = client_->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(coll_name).AddIndex(std::move(binary_idx)));
    milvus::test::ExpectStatusOK(status);
    status =
        client_->CreateIndex(milvus::CreateIndexRequest().WithCollectionName(coll_name).AddIndex(std::move(fp16_idx)));
    milvus::test::ExpectStatusOK(status);
    status = client_->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(coll_name).AddIndex(std::move(sparse_idx)));
    milvus::test::ExpectStatusOK(status);

    // prepare data
    size_t count = 100;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> float_gen{0.f, 1.f};
    std::uniform_int_distribution<uint8_t> byte_gen{0, 255};

    std::vector<std::vector<float>> float_vecs;
    std::vector<std::vector<uint8_t>> binary_vecs;
    std::vector<std::vector<float>> fp16_vecs;  // will be converted by the SDK
    std::vector<std::map<uint32_t, float>> sparse_vecs;

    for (size_t i = 0; i < count; ++i) {
        float_vecs.push_back({float_gen(rng), float_gen(rng), float_gen(rng), float_gen(rng)});
        binary_vecs.push_back({byte_gen(rng), byte_gen(rng), byte_gen(rng), byte_gen(rng)});  // 32 bits = 4 bytes
        fp16_vecs.push_back({float_gen(rng), float_gen(rng), float_gen(rng), float_gen(rng)});
        std::map<uint32_t, float> sparse;
        sparse[static_cast<uint32_t>(i)] = float_gen(rng);
        sparse[static_cast<uint32_t>(i + count)] = float_gen(rng);
        sparse_vecs.push_back(sparse);
    }

    // convert fp16_vecs (float) to actual float16 representation
    std::vector<std::vector<uint16_t>> fp16_data;
    for (const auto& vec : fp16_vecs) {
        fp16_data.push_back(milvus::ArrayF32toF16(vec));
    }

    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::FloatVecFieldData>("float_vec", float_vecs),
        std::make_shared<milvus::BinaryVecFieldData>("binary_vec", binary_vecs),
        std::make_shared<milvus::Float16VecFieldData>("fp16_vec", fp16_data),
        std::make_shared<milvus::SparseFloatVecFieldData>("sparse_vec", sparse_vecs)};

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(coll_name).WithColumnsData(std::move(fields));
    milvus::InsertResponse insert_resp;
    status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), count);

    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(coll_name));
    milvus::test::ExpectStatusOK(status);

    // search on float_vec
    {
        milvus::SearchRequest req;
        req.WithCollectionName(coll_name).WithAnnsField("float_vec").WithLimit(10);
        req.AddFloatVector({0.5f, 0.5f, 0.5f, 0.5f});
        req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::SearchResponse resp;
        status = client_->Search(req, resp);
        milvus::test::ExpectStatusOK(status);
        EXPECT_EQ(resp.Results().Results().size(), 1);
        EXPECT_EQ(resp.Results().Results().at(0).Scores().size(), 10);
    }

    // search on binary_vec
    {
        milvus::SearchRequest req;
        req.WithCollectionName(coll_name).WithAnnsField("binary_vec").WithLimit(10);
        req.WithMetricType(milvus::MetricType::HAMMING);
        req.AddBinaryVector(std::vector<uint8_t>{255, 255, 255, 255});
        req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::SearchResponse resp;
        status = client_->Search(req, resp);
        milvus::test::ExpectStatusOK(status);
        EXPECT_EQ(resp.Results().Results().size(), 1);
        EXPECT_EQ(resp.Results().Results().at(0).Scores().size(), 10);
    }

    // search on fp16_vec (pass float, SDK converts to fp16)
    {
        milvus::SearchRequest req;
        req.WithCollectionName(coll_name).WithAnnsField("fp16_vec").WithLimit(10);
        req.AddFloat16Vector(std::vector<float>{0.5f, 0.5f, 0.5f, 0.5f});
        req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::SearchResponse resp;
        status = client_->Search(req, resp);
        milvus::test::ExpectStatusOK(status);
        EXPECT_EQ(resp.Results().Results().size(), 1);
        EXPECT_EQ(resp.Results().Results().at(0).Scores().size(), 10);
    }

    // search on sparse_vec
    {
        milvus::SearchRequest req;
        req.WithCollectionName(coll_name).WithAnnsField("sparse_vec").WithLimit(10);
        req.WithMetricType(milvus::MetricType::IP);
        std::map<uint32_t, float> query_sparse = {{0, 1.0f}, {1, 0.5f}};
        req.AddSparseVector(query_sparse);
        req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::SearchResponse resp;
        status = client_->Search(req, resp);
        milvus::test::ExpectStatusOK(status);
        EXPECT_EQ(resp.Results().Results().size(), 1);
        EXPECT_GE(resp.Results().Results().at(0).Scores().size(), 1);
    }

    client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(coll_name));
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

TEST_F(MilvusServerTestSearch, HybridSearch) {
    // create a collection with 2 vector fields for hybrid search
    std::string coll_name = milvus::test::RanName("Hybrid_");
    auto schema = std::make_shared<milvus::CollectionSchema>(coll_name);
    schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
    schema->AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
    schema->AddField(milvus::FieldSchema("vec1", milvus::DataType::FLOAT_VECTOR, "vector 1").WithDimension(4));
    schema->AddField(milvus::FieldSchema("vec2", milvus::DataType::FLOAT_VECTOR, "vector 2").WithDimension(4));

    client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(coll_name));
    auto status = client_->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(coll_name).WithCollectionSchema(schema));
    milvus::test::ExpectStatusOK(status);

    // create indexes
    milvus::IndexDesc idx1("vec1", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    milvus::IndexDesc idx2("vec2", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    status = client_->CreateIndex(milvus::CreateIndexRequest().WithCollectionName(coll_name).AddIndex(std::move(idx1)));
    milvus::test::ExpectStatusOK(status);
    status = client_->CreateIndex(milvus::CreateIndexRequest().WithCollectionName(coll_name).AddIndex(std::move(idx2)));
    milvus::test::ExpectStatusOK(status);

    // insert data
    std::mt19937 rng(42);
    std::uniform_int_distribution<int16_t> age_gen{10, 50};
    std::uniform_real_distribution<float> vec_gen{0.f, 1.f};
    size_t count = 200;

    std::vector<int16_t> ages;
    std::vector<std::vector<float>> vecs1, vecs2;
    for (size_t i = 0; i < count; ++i) {
        ages.push_back(age_gen(rng));
        vecs1.push_back({vec_gen(rng), vec_gen(rng), vec_gen(rng), vec_gen(rng)});
        vecs2.push_back({vec_gen(rng), vec_gen(rng), vec_gen(rng), vec_gen(rng)});
    }

    std::vector<milvus::FieldDataPtr> fields{std::make_shared<milvus::Int16FieldData>("age", ages),
                                             std::make_shared<milvus::FloatVecFieldData>("vec1", vecs1),
                                             std::make_shared<milvus::FloatVecFieldData>("vec2", vecs2)};

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(coll_name).WithColumnsData(std::move(fields));
    milvus::InsertResponse insert_resp;
    status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);

    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(coll_name));
    milvus::test::ExpectStatusOK(status);

    // build sub search requests
    auto sub1 = std::make_shared<milvus::SubSearchRequest>();
    sub1->WithAnnsField("vec1").WithLimit(10).WithMetricType(milvus::MetricType::L2);
    sub1->AddFloatVector({0.5f, 0.5f, 0.5f, 0.5f});

    auto sub2 = std::make_shared<milvus::SubSearchRequest>();
    sub2->WithAnnsField("vec2").WithLimit(10).WithMetricType(milvus::MetricType::L2);
    sub2->AddFloatVector({0.5f, 0.5f, 0.5f, 0.5f});

    // hybrid search with RRF rerank
    milvus::HybridSearchRequest hybrid_req;
    hybrid_req.WithCollectionName(coll_name);
    hybrid_req.AddSubRequest(sub1);
    hybrid_req.AddSubRequest(sub2);
    hybrid_req.WithRerank(std::make_shared<milvus::RRFRerank>(60));
    hybrid_req.WithLimit(10);
    hybrid_req.AddOutputField("age");
    hybrid_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::HybridSearchResponse hybrid_resp;
    status = client_->HybridSearch(hybrid_req, hybrid_resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = hybrid_resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results.at(0).Scores().size(), 10);

    // hybrid search with Weighted rerank
    milvus::HybridSearchRequest hybrid_req2;
    hybrid_req2.WithCollectionName(coll_name);
    hybrid_req2.AddSubRequest(sub1);
    hybrid_req2.AddSubRequest(sub2);
    hybrid_req2.WithRerank(std::make_shared<milvus::WeightedRerank>(std::vector<float>{0.7f, 0.3f}));
    hybrid_req2.WithLimit(5);
    hybrid_req2.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::HybridSearchResponse hybrid_resp2;
    status = client_->HybridSearch(hybrid_req2, hybrid_resp2);
    milvus::test::ExpectStatusOK(status);

    const auto& results2 = hybrid_resp2.Results().Results();
    EXPECT_EQ(results2.size(), 1);
    EXPECT_EQ(results2.at(0).Scores().size(), 5);

    client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(coll_name));
}

TEST_F(MilvusServerTestSearch, ListQuerySegments) {
    createCollectionAndPartitions(true);

    // insert data
    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::Int16FieldData>("age", std::vector<int16_t>{12, 13}),
        std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"Tom", "Jerry"}),
        std::make_shared<milvus::FloatVecFieldData>(
            "face", std::vector<std::vector<float>>{{0.1f, 0.2f, 0.3f, 0.4f}, {0.5f, 0.6f, 0.7f, 0.8f}})};
    insertRecords(fields);
    loadCollection();

    // list query segments from loaded collection
    milvus::ListQuerySegmentsResponse seg_resp;
    auto status =
        client_->ListQuerySegments(milvus::ListQuerySegmentsRequest().WithCollectionName(collection_name), seg_resp);
    milvus::test::ExpectStatusOK(status);

    dropCollection();
}
