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

using testing::UnorderedElementsAre;

class MilvusServerTestQuery : public ::testing::Test {
 protected:
    static std::shared_ptr<milvus::MilvusClientV2> client_;
    static std::string collection_name;
    static std::vector<int64_t> inserted_ids;

    static void
    SetUpTestSuite() {
        const char* host = std::getenv("MILVUS_HOST");
        milvus::ConnectParam connect_param{host ? host : "localhost", 19530};
        client_ = milvus::MilvusClientV2::Create();
        auto status = client_->Connect(connect_param);
        milvus::test::ExpectStatusOK(status);

        collection_name = milvus::test::RanName("QueryTest_");

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

        // insert test data using row-based insert
        milvus::EntityRows rows_data;
        rows_data.push_back(nlohmann::json{{"age", 20}, {"name", "Alice"}, {"vec", {0.1f, 0.2f, 0.3f, 0.4f}}});
        rows_data.push_back(nlohmann::json{{"age", 25}, {"name", "Bob"}, {"vec", {0.2f, 0.3f, 0.4f, 0.5f}}});
        rows_data.push_back(nlohmann::json{{"age", 30}, {"name", "Charlie"}, {"vec", {0.3f, 0.4f, 0.5f, 0.6f}}});
        rows_data.push_back(nlohmann::json{{"age", 35}, {"name", "David"}, {"vec", {0.4f, 0.5f, 0.6f, 0.7f}}});
        rows_data.push_back(nlohmann::json{{"age", 40}, {"name", "Eve"}, {"vec", {0.5f, 0.6f, 0.7f, 0.8f}}});

        milvus::InsertRequest insert_req;
        insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
        milvus::InsertResponse insert_resp;
        status = client_->Insert(insert_req, insert_resp);
        milvus::test::ExpectStatusOK(status);
        inserted_ids = insert_resp.Results().IdArray().IntIDArray();

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

std::shared_ptr<milvus::MilvusClientV2> MilvusServerTestQuery::client_;
std::string MilvusServerTestQuery::collection_name;
std::vector<int64_t> MilvusServerTestQuery::inserted_ids;

TEST_F(MilvusServerTestQuery, QueryAll) {
    milvus::QueryRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter("age >= 0");
    req.AddOutputField("age");
    req.AddOutputField("name");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

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
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

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
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

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
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

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
    req2.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

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
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::GetResponse resp;
    auto status = client_->Get(req, resp);
    milvus::test::ExpectStatusOK(status);

    EXPECT_EQ(resp.Results().GetRowCount(), 3);

    auto age_field = resp.Results().OutputField<milvus::Int16FieldData>("age");
    ASSERT_NE(age_field, nullptr);
    EXPECT_EQ(age_field->Data().size(), 3);
}

TEST_F(MilvusServerTestQuery, QueryWithFilterTemplate) {
    // filter template with numeric value
    milvus::QueryRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter("age > {min_age}");
    req.AddFilterTemplate("min_age", 25);
    req.AddOutputField("age");
    req.AddOutputField("name");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::QueryResponse resp;
    auto status = client_->Query(req, resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(resp.Results().GetRowCount(), 3);

    // filter template with array (IN operator)
    milvus::QueryRequest req2;
    req2.WithCollectionName(collection_name);
    req2.WithFilter("name in {target_names}");
    req2.AddFilterTemplate("target_names", nlohmann::json{"Alice", "Charlie"});
    req2.AddOutputField("name");
    req2.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::QueryResponse resp2;
    status = client_->Query(req2, resp2);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(resp2.Results().GetRowCount(), 2);

    milvus::EntityRows rows;
    resp2.Results().OutputRows(rows);
    std::set<std::string> names;
    for (const auto& row : rows) {
        names.insert(row["name"].get<std::string>());
    }
    EXPECT_TRUE(names.count("Alice"));
    EXPECT_TRUE(names.count("Charlie"));

    // filter template with id list
    milvus::QueryRequest req3;
    req3.WithCollectionName(collection_name);
    req3.WithFilter("id in {target_ids}");
    req3.AddFilterTemplate("target_ids", nlohmann::json{inserted_ids[0], inserted_ids[1]});
    req3.AddOutputField("age");
    req3.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::QueryResponse resp3;
    status = client_->Query(req3, resp3);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(resp3.Results().GetRowCount(), 2);
}

TEST_F(MilvusServerTestQuery, GetByNonExistentIDs) {
    std::vector<int64_t> fake_ids{999999, 999998};

    milvus::GetRequest req;
    req.WithCollectionName(collection_name);
    req.WithIDs(std::move(fake_ids));
    req.AddOutputField("age");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::GetResponse resp;
    auto status = client_->Get(req, resp);
    milvus::test::ExpectStatusOK(status);

    EXPECT_EQ(resp.Results().GetRowCount(), 0);
}

///////////////////////////////////////////////////////////////////////////////
// String ID collection tests
///////////////////////////////////////////////////////////////////////////////
class MilvusServerTestQueryStringId : public ::testing::Test {
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

        collection_name = milvus::test::RanName("StrIdQuery_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        // VARCHAR primary key, no auto-id
        schema->AddField(
            milvus::FieldSchema("id", milvus::DataType::VARCHAR, "string id", true, false).WithMaxLength(64));
        schema->AddField(milvus::FieldSchema("score", milvus::DataType::INT32, "score"));
        schema->AddField(milvus::FieldSchema("tag", milvus::DataType::VARCHAR, "tag").WithMaxLength(64));
        schema->AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(4));

        status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        milvus::IndexDesc index_desc("vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
        status = client_->CreateIndex(
            milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
        milvus::test::ExpectStatusOK(status);

        // insert test data with string IDs
        milvus::EntityRows rows_data;
        rows_data.push_back(
            nlohmann::json{{"id", "apple"}, {"score", 90}, {"tag", "fruit"}, {"vec", {0.1f, 0.2f, 0.3f, 0.4f}}});
        rows_data.push_back(
            nlohmann::json{{"id", "banana"}, {"score", 80}, {"tag", "fruit"}, {"vec", {0.2f, 0.3f, 0.4f, 0.5f}}});
        rows_data.push_back(
            nlohmann::json{{"id", "carrot"}, {"score", 70}, {"tag", "vegetable"}, {"vec", {0.3f, 0.4f, 0.5f, 0.6f}}});
        rows_data.push_back(
            nlohmann::json{{"id", "dog"}, {"score", 95}, {"tag", "animal"}, {"vec", {0.4f, 0.5f, 0.6f, 0.7f}}});
        rows_data.push_back(
            nlohmann::json{{"id", "eagle"}, {"score", 85}, {"tag", "animal"}, {"vec", {0.5f, 0.6f, 0.7f, 0.8f}}});

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

std::shared_ptr<milvus::MilvusClientV2> MilvusServerTestQueryStringId::client_;
std::string MilvusServerTestQueryStringId::collection_name;

TEST_F(MilvusServerTestQueryStringId, QueryAll) {
    milvus::QueryRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter("score >= 0");
    req.AddOutputField("id");
    req.AddOutputField("score");
    req.AddOutputField("tag");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);

    milvus::QueryResponse resp;
    auto status = client_->Query(req, resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(resp.Results().GetRowCount(), 5);
}

TEST_F(MilvusServerTestQueryStringId, QueryWithFilter) {
    milvus::QueryRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter(R"(tag == "animal")");
    req.AddOutputField("id");
    req.AddOutputField("score");
    req.AddOutputField("tag");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);

    milvus::QueryResponse resp;
    auto status = client_->Query(req, resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(resp.Results().GetRowCount(), 2);

    milvus::EntityRows rows;
    resp.Results().OutputRows(rows);
    for (const auto& row : rows) {
        EXPECT_EQ(row["tag"].get<std::string>(), "animal");
        auto id = row["id"].get<std::string>();
        EXPECT_TRUE(id == "dog" || id == "eagle");
    }
}

TEST_F(MilvusServerTestQueryStringId, QueryWithStringIdFilter) {
    milvus::QueryRequest req;
    req.WithCollectionName(collection_name);
    req.WithFilter(R"(id in ["apple", "carrot", "eagle"])");
    req.AddOutputField("score");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);

    milvus::QueryResponse resp;
    auto status = client_->Query(req, resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(resp.Results().GetRowCount(), 3);
}

TEST_F(MilvusServerTestQueryStringId, GetByStringIDs) {
    std::vector<std::string> query_ids{"apple", "dog"};

    milvus::GetRequest req;
    req.WithCollectionName(collection_name);
    req.WithIDs(std::move(query_ids));
    req.AddOutputField("score");
    req.AddOutputField("tag");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);

    milvus::GetResponse resp;
    auto status = client_->Get(req, resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(resp.Results().GetRowCount(), 2);

    milvus::EntityRows rows;
    resp.Results().OutputRows(rows);
    std::map<std::string, nlohmann::json> result_map;
    for (const auto& row : rows) {
        result_map[row["id"].get<std::string>()] = row;
    }

    EXPECT_EQ(result_map["apple"]["score"].get<int32_t>(), 90);
    EXPECT_EQ(result_map["apple"]["tag"].get<std::string>(), "fruit");
    EXPECT_EQ(result_map["dog"]["score"].get<int32_t>(), 95);
    EXPECT_EQ(result_map["dog"]["tag"].get<std::string>(), "animal");
}

TEST_F(MilvusServerTestQueryStringId, GetByNonExistentStringIDs) {
    std::vector<std::string> fake_ids{"zzz_not_exist", "xxx_not_exist"};

    milvus::GetRequest req;
    req.WithCollectionName(collection_name);
    req.WithIDs(std::move(fake_ids));
    req.AddOutputField("score");
    req.WithConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);

    milvus::GetResponse resp;
    auto status = client_->Get(req, resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(resp.Results().GetRowCount(), 0);
}
