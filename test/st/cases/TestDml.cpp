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

#include "MilvusServerTest.h"

using milvus::test::MilvusServerTest;

class MilvusServerTestDml : public MilvusServerTest {
 protected:
    std::string collection_name;

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        collection_name = milvus::test::RanName("DmlTest_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        schema->AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
        schema->AddField(milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name").WithMaxLength(64));
        schema->AddField(milvus::FieldSchema("face", milvus::DataType::FLOAT_VECTOR, "face vector").WithDimension(4));

        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        milvus::IndexDesc index_desc("face", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
        status = client_->CreateIndex(
            milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
        milvus::test::ExpectStatusOK(status);
    }

    void
    TearDown() override {
        client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
        MilvusServerTest::TearDown();
    }
};

TEST_F(MilvusServerTestDml, InsertAndQuery) {
    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::Int16FieldData>("age", std::vector<int16_t>{20, 25, 30}),
        std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"Alice", "Bob", "Charlie"}),
        std::make_shared<milvus::FloatVecFieldData>(
            "face", std::vector<std::vector<float>>{
                        {0.1f, 0.2f, 0.3f, 0.4f}, {0.5f, 0.6f, 0.7f, 0.8f}, {0.9f, 1.0f, 1.1f, 1.2f}})};

    milvus::InsertRequest insert_req;
    std::vector<milvus::FieldDataPtr> fields_copy(fields);
    insert_req.WithCollectionName(collection_name).WithColumnsData(std::move(fields_copy));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 3);

    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("age > 20");
    query_req.AddOutputField("age");
    query_req.AddOutputField("name");
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    query_resp.Results().OutputRows(rows);
    EXPECT_EQ(rows.size(), 2);
    for (const auto& row : rows) {
        EXPECT_GT(row["age"].get<int16_t>(), 20);
        auto name = row["name"].get<std::string>();
        EXPECT_TRUE(name == "Bob" || name == "Charlie");
    }
}

TEST_F(MilvusServerTestDml, DeleteByFilter) {
    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::Int16FieldData>("age", std::vector<int16_t>{20, 25}),
        std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"Alice", "Bob"}),
        std::make_shared<milvus::FloatVecFieldData>(
            "face", std::vector<std::vector<float>>{{0.1f, 0.2f, 0.3f, 0.4f}, {0.5f, 0.6f, 0.7f, 0.8f}})};

    milvus::InsertRequest insert_req;
    std::vector<milvus::FieldDataPtr> fields_copy(fields);
    insert_req.WithCollectionName(collection_name).WithColumnsData(std::move(fields_copy));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);

    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    milvus::DeleteRequest delete_req;
    delete_req.WithCollectionName(collection_name).WithFilter("age == 20");
    milvus::DeleteResponse delete_resp;
    status = client_->Delete(delete_req, delete_resp);
    milvus::test::ExpectStatusOK(status);

    // query to verify deletion
    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("age < 25");
    query_req.AddOutputField("age");
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    query_resp.Results().OutputRows(rows);
    EXPECT_EQ(rows.size(), 0);
}

TEST_F(MilvusServerTestDml, Upsert) {
    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::Int16FieldData>("age", std::vector<int16_t>{20, 25}),
        std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"Alice", "Bob"}),
        std::make_shared<milvus::FloatVecFieldData>(
            "face", std::vector<std::vector<float>>{{0.1f, 0.2f, 0.3f, 0.4f}, {0.5f, 0.6f, 0.7f, 0.8f}})};

    milvus::InsertRequest insert_req;
    std::vector<milvus::FieldDataPtr> fields_copy(fields);
    insert_req.WithCollectionName(collection_name).WithColumnsData(std::move(fields_copy));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);

    // upsert with updated data
    auto ids = insert_resp.Results().IdArray().IntIDArray();
    std::vector<milvus::FieldDataPtr> upsert_fields{
        std::make_shared<milvus::Int64FieldData>("id", ids),
        std::make_shared<milvus::Int16FieldData>("age", std::vector<int16_t>{21, 26}),
        std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"Alice_v2", "Bob_v2"}),
        std::make_shared<milvus::FloatVecFieldData>(
            "face", std::vector<std::vector<float>>{{0.2f, 0.3f, 0.4f, 0.5f}, {0.6f, 0.7f, 0.8f, 0.9f}})};

    milvus::UpsertRequest upsert_req;
    upsert_req.WithCollectionName(collection_name).WithColumnsData(std::move(upsert_fields));
    milvus::UpsertResponse upsert_resp;
    status = client_->Upsert(upsert_req, upsert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(upsert_resp.Results().UpsertCount(), 2);
    EXPECT_EQ(upsert_resp.Results().IdArray().IntIDArray().size(), 2);

    // after upsert, milvus will generate a new id for each upserted entity, so the returned ids should be different
    // from original ones
    EXPECT_NE(upsert_resp.Results().IdArray().IntIDArray()[0], ids[0]);
    EXPECT_NE(upsert_resp.Results().IdArray().IntIDArray()[1], ids[1]);
}

TEST_F(MilvusServerTestDml, ListPersistentSegments) {
    // insert data
    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::Int16FieldData>("age", std::vector<int16_t>{20, 25}),
        std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"Alice", "Bob"}),
        std::make_shared<milvus::FloatVecFieldData>(
            "face", std::vector<std::vector<float>>{{0.1f, 0.2f, 0.3f, 0.4f}, {0.5f, 0.6f, 0.7f, 0.8f}})};

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithColumnsData(std::move(fields));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);

    // flush to create persistent segments
    status = client_->Flush(milvus::FlushRequest().AddCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    // list persistent segments
    milvus::ListPersistentSegmentsResponse seg_resp;
    status = client_->ListPersistentSegments(
        milvus::ListPersistentSegmentsRequest().WithCollectionName(collection_name), seg_resp);
    milvus::test::ExpectStatusOK(status);

    EXPECT_GT(seg_resp.Result().size(), 0);
}
