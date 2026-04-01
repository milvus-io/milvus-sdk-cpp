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
using testing::UnorderedElementsAre;

class MilvusServerTestQuery : public MilvusServerTest {
 protected:
    std::string collection_name;
    std::vector<int64_t> inserted_ids;

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        collection_name = milvus::test::RanName("QueryTest_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        schema->AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
        schema->AddField(milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name").WithMaxLength(64));
        schema->AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(4));

        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        milvus::IndexDesc index_desc("vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
        status = client_->CreateIndex(
            milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
        milvus::test::ExpectStatusOK(status);

        // insert test data
        std::vector<milvus::FieldDataPtr> fields{
            std::make_shared<milvus::Int16FieldData>("age", std::vector<int16_t>{20, 25, 30, 35, 40}),
            std::make_shared<milvus::VarCharFieldData>(
                "name", std::vector<std::string>{"Alice", "Bob", "Charlie", "David", "Eve"}),
            std::make_shared<milvus::FloatVecFieldData>("vec",
                                                        std::vector<std::vector<float>>{{0.1f, 0.2f, 0.3f, 0.4f},
                                                                                        {0.2f, 0.3f, 0.4f, 0.5f},
                                                                                        {0.3f, 0.4f, 0.5f, 0.6f},
                                                                                        {0.4f, 0.5f, 0.6f, 0.7f},
                                                                                        {0.5f, 0.6f, 0.7f, 0.8f}})};

        milvus::InsertRequest insert_req;
        insert_req.WithCollectionName(collection_name).WithColumnsData(std::move(fields));
        milvus::InsertResponse insert_resp;
        status = client_->Insert(insert_req, insert_resp);
        milvus::test::ExpectStatusOK(status);
        inserted_ids = insert_resp.Results().IdArray().IntIDArray();

        status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
        milvus::test::ExpectStatusOK(status);
    }

    void
    TearDown() override {
        client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
        MilvusServerTest::TearDown();
    }
};

TEST_F(MilvusServerTestQuery, QueryAll) {
    milvus::QueryRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter("age >= 0");
    req.AddOutputField("age");
    req.AddOutputField("name");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse resp;
    auto status = client_->Query(req, resp);
    milvus::test::ExpectStatusOK(status);

    EXPECT_EQ(resp.Results().GetRowCount(), 5);

    auto age_field = resp.Results().OutputField<milvus::Int16FieldData>("age");
    ASSERT_NE(age_field, nullptr);
    EXPECT_THAT(age_field->Data(), UnorderedElementsAre(20, 25, 30, 35, 40));

    auto name_field = resp.Results().OutputField<milvus::VarCharFieldData>("name");
    ASSERT_NE(name_field, nullptr);
    EXPECT_THAT(name_field->Data(), UnorderedElementsAre("Alice", "Bob", "Charlie", "David", "Eve"));
}

TEST_F(MilvusServerTestQuery, QueryWithFilter) {
    milvus::QueryRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter("age > 25");
    req.AddOutputField("age");
    req.AddOutputField("name");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse resp;
    auto status = client_->Query(req, resp);
    milvus::test::ExpectStatusOK(status);

    EXPECT_EQ(resp.Results().GetRowCount(), 3);

    auto age_field = resp.Results().OutputField<milvus::Int16FieldData>("age");
    ASSERT_NE(age_field, nullptr);
    EXPECT_THAT(age_field->Data(), UnorderedElementsAre(30, 35, 40));
}

TEST_F(MilvusServerTestQuery, QueryWithStringFilter) {
    milvus::QueryRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter("name like \"A%\"");
    req.AddOutputField("name");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse resp;
    auto status = client_->Query(req, resp);
    milvus::test::ExpectStatusOK(status);

    EXPECT_EQ(resp.Results().GetRowCount(), 1);

    auto name_field = resp.Results().OutputField<milvus::VarCharFieldData>("name");
    ASSERT_NE(name_field, nullptr);
    EXPECT_EQ(name_field->Data(), std::vector<std::string>{"Alice"});
}

TEST_F(MilvusServerTestQuery, QueryWithLimitAndOffset) {
    milvus::QueryRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter("age >= 0");
    req.AddOutputField("age");
    req.WithLimit(2);
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse resp;
    auto status = client_->Query(req, resp);
    milvus::test::ExpectStatusOK(status);

    EXPECT_EQ(resp.Results().GetRowCount(), 2);

    // query with offset
    milvus::QueryRequest req2;
    req2.WithCollectionName(collection_name);
    req2.WithFilter("age >= 0");
    req2.AddOutputField("age");
    req2.WithLimit(2);
    req2.WithOffset(3);
    req2.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse resp2;
    status = client_->Query(req2, resp2);
    milvus::test::ExpectStatusOK(status);

    EXPECT_EQ(resp2.Results().GetRowCount(), 2);
}

TEST_F(MilvusServerTestQuery, GetByIDs) {
    // get first 3 inserted IDs
    std::vector<int64_t> query_ids(inserted_ids.begin(), inserted_ids.begin() + 3);

    milvus::GetRequest req;
    req.WithCollectionName(collection_name);
    req.WithIDs(std::move(query_ids));
    req.AddOutputField("age");
    req.AddOutputField("name");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::GetResponse resp;
    auto status = client_->Get(req, resp);
    milvus::test::ExpectStatusOK(status);

    EXPECT_EQ(resp.Results().GetRowCount(), 3);

    auto age_field = resp.Results().OutputField<milvus::Int16FieldData>("age");
    ASSERT_NE(age_field, nullptr);
    EXPECT_EQ(age_field->Data().size(), 3);
}

TEST_F(MilvusServerTestQuery, GetByNonExistentIDs) {
    std::vector<int64_t> fake_ids{999999, 999998};

    milvus::GetRequest req;
    req.WithCollectionName(collection_name);
    req.WithIDs(std::move(fake_ids));
    req.AddOutputField("age");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::GetResponse resp;
    auto status = client_->Get(req, resp);
    milvus::test::ExpectStatusOK(status);

    EXPECT_EQ(resp.Results().GetRowCount(), 0);
}
