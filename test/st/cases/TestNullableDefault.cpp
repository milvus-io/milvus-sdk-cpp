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

///////////////////////////////////////////////////////////////////////////////
// Nullable field tests
///////////////////////////////////////////////////////////////////////////////
class MilvusServerTestNullable : public MilvusServerTest {
 protected:
    std::string collection_name;
    static constexpr uint32_t dimension = 4;

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        collection_name = milvus::test::RanName("Nullable_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, false));
        schema->AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(dimension));
        schema->AddField(
            milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name").WithMaxLength(256).WithNullable(true));
        schema->AddField(milvus::FieldSchema("age", milvus::DataType::INT8, "age").WithNullable(true));
        schema->AddField(milvus::FieldSchema("scores", milvus::DataType::ARRAY, "scores")
                             .WithElementType(milvus::DataType::FLOAT)
                             .WithMaxCapacity(10)
                             .WithNullable(true));

        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        milvus::IndexDesc idx("vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
        status = client_->CreateIndex(
            milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(idx)));
        milvus::test::ExpectStatusOK(status);

        status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
        milvus::test::ExpectStatusOK(status);
    }

    void
    TearDown() override {
        client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
        MilvusServerTest::TearDown();
    }
};

TEST_F(MilvusServerTestNullable, InsertRowBasedWithNull) {
    milvus::EntityRows rows_data;
    // row 0: all fields have values
    rows_data.push_back(nlohmann::json{
        {"id", 0},
        {"vec", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"name", "Alice"},
        {"age", 25},
        {"scores", {1.0f, 2.0f, 3.0f}},
    });
    // row 1: nullable fields explicitly set to null
    rows_data.push_back(nlohmann::json{
        {"id", 1},
        {"vec", {0.5f, 0.6f, 0.7f, 0.8f}},
        {"name", nullptr},
        {"age", nullptr},
        {"scores", nullptr},
    });
    // row 2: nullable fields omitted (treated as null)
    rows_data.push_back(nlohmann::json{
        {"id", 2},
        {"vec", {0.9f, 1.0f, 1.1f, 1.2f}},
    });

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 3);

    // query rows where name is null
    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("name is null");
    query_req.AddOutputField("id");
    query_req.AddOutputField("name");
    query_req.AddOutputField("age");
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    query_resp.Results().OutputRows(rows);
    EXPECT_EQ(rows.size(), 2);  // rows 1 and 2
    for (const auto& row : rows) {
        EXPECT_TRUE(row["name"].is_null());
    }

    // query rows where name is not null
    milvus::QueryRequest query_req2;
    query_req2.WithCollectionName(collection_name);
    query_req2.WithFilter("name is not null");
    query_req2.AddOutputField("id");
    query_req2.AddOutputField("name");
    query_req2.AddOutputField("age");
    query_req2.AddOutputField("scores");
    query_req2.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::QueryResponse query_resp2;
    status = client_->Query(query_req2, query_resp2);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows2;
    query_resp2.Results().OutputRows(rows2);
    EXPECT_EQ(rows2.size(), 1);  // only row 0
    EXPECT_EQ(rows2[0]["id"].get<int64_t>(), 0);
    EXPECT_EQ(rows2[0]["name"].get<std::string>(), "Alice");
    EXPECT_EQ(rows2[0]["age"].get<int8_t>(), 25);
}

TEST_F(MilvusServerTestNullable, InsertColumnBasedWithNull) {
    auto id_field = std::make_shared<milvus::Int64FieldData>("id");
    auto vec_field = std::make_shared<milvus::FloatVecFieldData>("vec");
    auto name_field = std::make_shared<milvus::VarCharFieldData>("name");
    auto age_field = std::make_shared<milvus::Int8FieldData>("age");

    // row 0: values present
    id_field->Add(10);
    vec_field->Add({0.1f, 0.2f, 0.3f, 0.4f});
    name_field->Add("Bob");
    age_field->Add(30);

    // row 1: null values
    id_field->Add(11);
    vec_field->Add({0.5f, 0.6f, 0.7f, 0.8f});
    name_field->AddNull();
    age_field->AddNull();

    // row 2: null values
    id_field->Add(12);
    vec_field->Add({0.9f, 1.0f, 1.1f, 1.2f});
    name_field->AddNull();
    age_field->AddNull();

    std::vector<milvus::FieldDataPtr> fields{id_field, vec_field, name_field, age_field};
    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithColumnsData(std::move(fields));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 3);

    // query and verify via column-based output
    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("id >= 10");
    query_req.AddOutputField("name");
    query_req.AddOutputField("age");
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(query_resp.Results().GetRowCount(), 3);

    auto out_name = query_resp.Results().OutputField<milvus::VarCharFieldData>("name");
    auto out_age = query_resp.Results().OutputField<milvus::Int8FieldData>("age");
    ASSERT_NE(out_name, nullptr);
    ASSERT_NE(out_age, nullptr);

    // verify null flags — exactly 1 non-null and 2 null
    int null_count = 0;
    for (size_t i = 0; i < out_name->Count(); ++i) {
        if (out_name->IsNull(i)) {
            null_count++;
        }
    }
    EXPECT_EQ(null_count, 2);
}

TEST_F(MilvusServerTestNullable, SearchWithNullFilter) {
    // insert mixed null/non-null data
    milvus::EntityRows rows_data;
    rows_data.push_back(nlohmann::json{
        {"id", 0},
        {"vec", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"name", "Alice"},
        {"age", 20},
    });
    rows_data.push_back(nlohmann::json{
        {"id", 1},
        {"vec", {0.5f, 0.6f, 0.7f, 0.8f}},
        {"name", nullptr},
    });
    rows_data.push_back(nlohmann::json{
        {"id", 2},
        {"vec", {0.9f, 1.0f, 1.1f, 1.2f}},
        {"name", "Charlie"},
        {"age", 30},
    });

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);

    // search with filter "age is not null"
    milvus::SearchRequest search_req;
    search_req.WithCollectionName(collection_name);
    search_req.WithAnnsField("vec");
    search_req.WithLimit(10);
    search_req.AddFloatVector({0.5f, 0.5f, 0.5f, 0.5f});
    search_req.WithFilter("age is not null");
    search_req.AddOutputField("name");
    search_req.AddOutputField("age");
    search_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::SearchResponse search_resp;
    status = client_->Search(search_req, search_resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = search_resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results.at(0).Scores().size(), 2);  // only rows 0 and 2

    milvus::EntityRows rows;
    results.at(0).OutputRows(rows);
    for (const auto& row : rows) {
        EXPECT_FALSE(row["name"].is_null());
        EXPECT_FALSE(row["age"].is_null());
    }
}

///////////////////////////////////////////////////////////////////////////////
// Default value tests
///////////////////////////////////////////////////////////////////////////////
class MilvusServerTestDefaultValue : public MilvusServerTest {
 protected:
    std::string collection_name;
    static constexpr uint32_t dimension = 4;

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        collection_name = milvus::test::RanName("Default_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, false));
        schema->AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(dimension));
        schema->AddField(milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name")
                             .WithMaxLength(256)
                             .WithDefaultValue("Unknown"));
        schema->AddField(milvus::FieldSchema("price", milvus::DataType::FLOAT, "price").WithDefaultValue(9.99));
        schema->AddField(milvus::FieldSchema("count", milvus::DataType::INT32, "count").WithDefaultValue(0));

        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        milvus::IndexDesc idx("vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
        status = client_->CreateIndex(
            milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(idx)));
        milvus::test::ExpectStatusOK(status);

        status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
        milvus::test::ExpectStatusOK(status);
    }

    void
    TearDown() override {
        client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
        MilvusServerTest::TearDown();
    }
};

TEST_F(MilvusServerTestDefaultValue, InsertRowBasedWithDefault) {
    milvus::EntityRows rows_data;
    // row 0: all fields provided
    rows_data.push_back(nlohmann::json{
        {"id", 0},
        {"vec", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"name", "Alice"},
        {"price", 19.99f},
        {"count", 5},
    });
    // row 1: omit fields with defaults — should get default values
    rows_data.push_back(nlohmann::json{
        {"id", 1},
        {"vec", {0.5f, 0.6f, 0.7f, 0.8f}},
    });
    // row 2: only provide some fields
    rows_data.push_back(nlohmann::json{
        {"id", 2},
        {"vec", {0.9f, 1.0f, 1.1f, 1.2f}},
        {"name", "Charlie"},
    });

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 3);

    // query all and verify default values
    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("id >= 0");
    query_req.AddOutputField("id");
    query_req.AddOutputField("name");
    query_req.AddOutputField("price");
    query_req.AddOutputField("count");
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    query_resp.Results().OutputRows(rows);
    EXPECT_EQ(rows.size(), 3);

    // build lookup by id
    std::map<int64_t, nlohmann::json> result_map;
    for (const auto& row : rows) {
        result_map[row["id"].get<int64_t>()] = row;
    }

    // row 0: explicit values
    EXPECT_EQ(result_map[0]["name"].get<std::string>(), "Alice");
    EXPECT_FLOAT_EQ(result_map[0]["price"].get<float>(), 19.99f);
    EXPECT_EQ(result_map[0]["count"].get<int32_t>(), 5);

    // row 1: all defaults
    EXPECT_EQ(result_map[1]["name"].get<std::string>(), "Unknown");
    EXPECT_FLOAT_EQ(result_map[1]["price"].get<float>(), 9.99f);
    EXPECT_EQ(result_map[1]["count"].get<int32_t>(), 0);

    // row 2: name explicit, price and count default
    EXPECT_EQ(result_map[2]["name"].get<std::string>(), "Charlie");
    EXPECT_FLOAT_EQ(result_map[2]["price"].get<float>(), 9.99f);
    EXPECT_EQ(result_map[2]["count"].get<int32_t>(), 0);
}

TEST_F(MilvusServerTestDefaultValue, InsertColumnBasedOmitDefaultField) {
    // insert without providing "price" and "count" columns — they should get defaults
    auto id_field = std::make_shared<milvus::Int64FieldData>("id");
    auto vec_field = std::make_shared<milvus::FloatVecFieldData>("vec");
    auto name_field = std::make_shared<milvus::VarCharFieldData>("name");

    id_field->Add(10);
    vec_field->Add({0.1f, 0.2f, 0.3f, 0.4f});
    name_field->Add("David");

    id_field->Add(11);
    vec_field->Add({0.5f, 0.6f, 0.7f, 0.8f});
    name_field->Add("Eve");

    std::vector<milvus::FieldDataPtr> fields{id_field, vec_field, name_field};
    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithColumnsData(std::move(fields));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 2);

    // query and verify defaults
    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("id >= 10");
    query_req.AddOutputField("name");
    query_req.AddOutputField("price");
    query_req.AddOutputField("count");
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    query_resp.Results().OutputRows(rows);
    EXPECT_EQ(rows.size(), 2);
    for (const auto& row : rows) {
        EXPECT_FLOAT_EQ(row["price"].get<float>(), 9.99f);
        EXPECT_EQ(row["count"].get<int32_t>(), 0);
    }
}

TEST_F(MilvusServerTestDefaultValue, SearchFilterOnDefaultValue) {
    milvus::EntityRows rows_data;
    // row with explicit name
    rows_data.push_back(nlohmann::json{
        {"id", 0},
        {"vec", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"name", "Alice"},
        {"price", 5.0f},
    });
    // row with default name "Unknown"
    rows_data.push_back(nlohmann::json{
        {"id", 1},
        {"vec", {0.5f, 0.6f, 0.7f, 0.8f}},
    });
    // row with default name "Unknown"
    rows_data.push_back(nlohmann::json{
        {"id", 2},
        {"vec", {0.9f, 1.0f, 1.1f, 1.2f}},
    });

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);

    // search with filter excluding default name
    milvus::SearchRequest search_req;
    search_req.WithCollectionName(collection_name);
    search_req.WithAnnsField("vec");
    search_req.WithLimit(10);
    search_req.AddFloatVector({0.5f, 0.5f, 0.5f, 0.5f});
    search_req.WithFilter(R"(name != "Unknown")");
    search_req.AddOutputField("name");
    search_req.AddOutputField("price");
    search_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::SearchResponse search_resp;
    status = client_->Search(search_req, search_resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = search_resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results.at(0).Scores().size(), 1);  // only Alice

    milvus::EntityRows rows;
    results.at(0).OutputRows(rows);
    EXPECT_EQ(rows[0]["name"].get<std::string>(), "Alice");
    EXPECT_FLOAT_EQ(rows[0]["price"].get<float>(), 5.0f);
}
