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

class MilvusServerTestStruct : public MilvusServerTest {
 protected:
    std::string collection_name;
    static constexpr uint32_t dimension = 4;

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        collection_name = milvus::test::RanName("StructTest_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, false));
        schema->AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(dimension));

        // struct field with scalar and vector sub-fields
        milvus::StructFieldSchema struct_schema =
            milvus::StructFieldSchema()
                .WithName("st")
                .WithMaxCapacity(10)
                .AddField(milvus::FieldSchema("st_int32", milvus::DataType::INT32))
                .AddField(milvus::FieldSchema("st_varchar", milvus::DataType::VARCHAR).WithMaxLength(256))
                .AddField(milvus::FieldSchema("st_vec", milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
        schema->AddStructField(std::move(struct_schema));

        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        // create indexes
        milvus::IndexParam idx_vec("vec", "", milvus::IndexType::FLAT, milvus::MetricType::COSINE);
        milvus::IndexParam idx_st_vec("st[st_vec]", "", milvus::IndexType::HNSW, milvus::MetricType::MAX_SIM_COSINE);
        status = client_->CreateIndex(milvus::CreateIndexRequest()
                                          .WithCollectionName(collection_name)
                                          .AddIndexParam(std::move(idx_vec))
                                          .AddIndexParam(std::move(idx_st_vec)));
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

TEST_F(MilvusServerTestStruct, InsertAndQueryRowBased) {
    // insert using row-based with struct field
    milvus::EntityRows rows_data;
    for (int i = 0; i < 5; ++i) {
        milvus::EntityRow row;
        row["id"] = i;
        row["vec"] = std::vector<float>{0.1f * (i + 1), 0.2f * (i + 1), 0.3f * (i + 1), 0.4f * (i + 1)};

        // each row has (i+1) structs in the list
        std::vector<nlohmann::json> struct_list;
        for (int k = 0; k <= i; ++k) {
            nlohmann::json st;
            st["st_int32"] = k * 10;
            st["st_varchar"] = "row_" + std::to_string(i) + "_item_" + std::to_string(k);
            st["st_vec"] = std::vector<float>{0.1f * (k + 1), 0.2f * (k + 1), 0.3f * (k + 1), 0.4f * (k + 1)};
            struct_list.emplace_back(std::move(st));
        }
        row["st"] = struct_list;
        rows_data.emplace_back(std::move(row));
    }

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 5);

    // query all rows and output the struct field
    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("id >= 0");
    query_req.AddOutputField("id");
    query_req.AddOutputField("st");
    query_req.WithLimit(100);
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    status = query_resp.Results().OutputRows(rows);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(rows.size(), 5);

    for (const auto& row : rows) {
        EXPECT_TRUE(row.contains("id"));
        EXPECT_TRUE(row.contains("st"));
        auto id = row["id"].get<int64_t>();

        // each row should have (id+1) struct elements
        const auto& st_list = row["st"];
        EXPECT_EQ(st_list.size(), id + 1);

        // verify each struct element's scalar values
        for (int k = 0; k <= id; ++k) {
            const auto& st = st_list[k];
            EXPECT_EQ(st["st_int32"].get<int32_t>(), k * 10);
            EXPECT_EQ(st["st_varchar"].get<std::string>(), "row_" + std::to_string(id) + "_item_" + std::to_string(k));
        }
    }
}

TEST_F(MilvusServerTestStruct, InsertColumnBased) {
    // insert using column-based with StructFieldData
    std::vector<int64_t> ids;
    std::vector<std::vector<float>> vecs;
    std::vector<std::vector<nlohmann::json>> structs;

    for (int i = 0; i < 3; ++i) {
        ids.push_back(100 + i);
        vecs.push_back({0.1f * (i + 1), 0.2f * (i + 1), 0.3f * (i + 1), 0.4f * (i + 1)});

        std::vector<nlohmann::json> struct_list;
        for (int k = 0; k <= i; ++k) {
            nlohmann::json st;
            st["st_int32"] = k * 100;
            st["st_varchar"] = "col_" + std::to_string(i) + "_" + std::to_string(k);
            st["st_vec"] = std::vector<float>{0.5f, 0.5f, 0.5f, 0.5f};
            struct_list.emplace_back(std::move(st));
        }
        structs.emplace_back(std::move(struct_list));
    }

    std::vector<milvus::FieldDataPtr> fields{std::make_shared<milvus::Int64FieldData>("id", ids),
                                             std::make_shared<milvus::FloatVecFieldData>("vec", vecs),
                                             std::make_shared<milvus::StructFieldData>("st", structs)};

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithColumnsData(std::move(fields));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 3);

    // query and verify
    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("id >= 100");
    query_req.AddOutputField("id");
    query_req.AddOutputField("st");
    query_req.WithLimit(100);
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    status = query_resp.Results().OutputRows(rows);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(rows.size(), 3);

    for (const auto& row : rows) {
        auto id = row["id"].get<int64_t>();
        int i = static_cast<int>(id - 100);

        const auto& st_list = row["st"];
        EXPECT_EQ(st_list.size(), i + 1);

        for (int k = 0; k <= i; ++k) {
            const auto& st = st_list[k];
            EXPECT_EQ(st["st_int32"].get<int32_t>(), k * 100);
            EXPECT_EQ(st["st_varchar"].get<std::string>(), "col_" + std::to_string(i) + "_" + std::to_string(k));
        }
    }
}

TEST_F(MilvusServerTestStruct, SearchOnStructVectorField) {
    // insert data
    milvus::EntityRows rows_data;
    for (int i = 0; i < 100; ++i) {
        milvus::EntityRow row;
        row["id"] = i;
        row["vec"] = std::vector<float>{0.1f * (i % 10 + 1), 0.2f, 0.3f, 0.4f};

        std::vector<nlohmann::json> struct_list;
        nlohmann::json st;
        st["st_int32"] = i;
        st["st_varchar"] = "search_" + std::to_string(i);
        st["st_vec"] = std::vector<float>{0.1f * (i % 10 + 1), 0.2f, 0.3f, 0.4f};
        struct_list.emplace_back(std::move(st));
        row["st"] = struct_list;

        rows_data.emplace_back(std::move(row));
    }

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);

    // search on the struct's vector sub-field using EmbeddingList
    milvus::SearchRequest search_req;
    search_req.WithCollectionName(collection_name);
    search_req.WithAnnsField("st[st_vec]");
    search_req.WithLimit(5);
    search_req.AddOutputField("st[st_varchar]");
    search_req.AddOutputField("st[st_int32]");
    search_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::EmbeddingList emb_list;
    emb_list.AddFloatVector({0.5f, 0.2f, 0.3f, 0.4f});
    search_req.AddEmbeddingList(std::move(emb_list));

    milvus::SearchResponse search_resp;
    status = client_->Search(search_req, search_resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = search_resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results.at(0).Scores().size(), 5);

    // verify output fields match inserted data
    milvus::EntityRows search_rows;
    status = results.at(0).OutputRows(search_rows);
    milvus::test::ExpectStatusOK(status);
    for (const auto& row : search_rows) {
        auto id = row["id"].get<int64_t>();

        // st is an array of structs, search only return one element that most similar to the query embedding
        auto st_array = row["st"].get<std::vector<nlohmann::json>>();
        ASSERT_EQ(st_array.size(), 1);
        EXPECT_EQ(st_array[0]["st_int32"].get<int32_t>(), static_cast<int32_t>(id));
        EXPECT_EQ(st_array[0]["st_varchar"].get<std::string>(), "search_" + std::to_string(id));
    }
}

TEST_F(MilvusServerTestStruct, QueryStructSubField) {
    // insert data
    milvus::EntityRows rows_data;
    for (int i = 0; i < 5; ++i) {
        milvus::EntityRow row;
        row["id"] = i;
        row["vec"] = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f};

        nlohmann::json st;
        st["st_int32"] = i * 10;
        st["st_varchar"] = "item_" + std::to_string(i);
        st["st_vec"] = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f};
        row["st"] = std::vector<nlohmann::json>{st};

        rows_data.emplace_back(std::move(row));
    }

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);

    // query outputting specific struct sub-fields
    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("id >= 0");
    query_req.AddOutputField("id");
    query_req.AddOutputField("st[st_int32]");
    query_req.AddOutputField("st[st_varchar]");
    query_req.WithLimit(100);
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    status = query_resp.Results().OutputRows(rows);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(rows.size(), 5);

    for (const auto& row : rows) {
        auto id = row["id"].get<int64_t>();

        // st is an array of structs, search only return one element that most similar to the query embedding
        auto st_array = row["st"].get<std::vector<nlohmann::json>>();
        ASSERT_EQ(st_array.size(), 1);
        EXPECT_EQ(st_array[0]["st_int32"].get<int32_t>(), static_cast<int32_t>(id * 10));
        EXPECT_EQ(st_array[0]["st_varchar"].get<std::string>(), "item_" + std::to_string(id));
    }
}

class MilvusServerTestStructElementNull : public MilvusServerTest {
 protected:
    std::string collection_name;
    static constexpr uint32_t dimension = 4;

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        collection_name = milvus::test::RanName("StructElementNull_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, false));

        // element-nullable scalar/vector sub-fields and a nested Array sub-field
        milvus::StructFieldSchema struct_schema =
            milvus::StructFieldSchema()
                .WithName("st")
                .WithMaxCapacity(10)
                .WithNullable(true)
                .AddField(milvus::FieldSchema("st_int32", milvus::DataType::INT32).WithNullable(true))
                .AddField(
                    milvus::FieldSchema("st_varchar", milvus::DataType::VARCHAR).WithMaxLength(64).WithNullable(true))
                .AddField(milvus::FieldSchema("st_vec", milvus::DataType::FLOAT_VECTOR)
                              .WithDimension(dimension)
                              .WithNullable(true))
                .AddField(milvus::FieldSchema("st_tags", milvus::DataType::ARRAY)
                              .WithElementType(milvus::DataType::VARCHAR)
                              .WithMaxCapacity(8)
                              .WithMaxLength(16)
                              .WithNullable(true)
                              .WithElementNullable(true));
        schema->AddStructField(std::move(struct_schema));

        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        // element-nullable vector sub-fields use element-level metrics
        milvus::IndexParam idx_st_vec("st[st_vec]", "", milvus::IndexType::HNSW, milvus::MetricType::COSINE);
        status = client_->CreateIndex(
            milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndexParam(std::move(idx_st_vec)));
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

TEST_F(MilvusServerTestStructElementNull, InsertAndQueryElementNull) {
    // insert rows whose struct elements carry null scalar/vector sub-fields and null leaves
    milvus::EntityRows rows_data;
    for (int i = 0; i < 3; ++i) {
        milvus::EntityRow row;
        row["id"] = i;

        std::vector<nlohmann::json> struct_list;
        {
            nlohmann::json st;
            st["st_int32"] = i * 10;
            st["st_varchar"] = "item_" + std::to_string(i);
            st["st_vec"] = std::vector<float>{0.1f * (i + 1), 0.2f, 0.3f, 0.4f};
            nlohmann::json tags = nlohmann::json::array({"a", nullptr});
            st["st_tags"] = tags;
            struct_list.emplace_back(std::move(st));
        }
        {
            nlohmann::json st;
            st["st_int32"] = i * 10 + 1;
            st["st_varchar"] = nullptr;
            st["st_vec"] = nullptr;
            struct_list.emplace_back(std::move(st));
        }
        row["st"] = struct_list;
        rows_data.emplace_back(std::move(row));
    }

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 3);

    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("id >= 0");
    query_req.AddOutputField("id");
    query_req.AddOutputField("st");
    query_req.WithLimit(100);
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    status = query_resp.Results().OutputRows(rows);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(rows.size(), 3);

    for (const auto& row : rows) {
        auto id = row["id"].get<int64_t>();
        const auto& st_list = row["st"];
        ASSERT_EQ(st_list.size(), 2u);
        EXPECT_EQ(st_list[0]["st_int32"].get<int32_t>(), static_cast<int32_t>(id * 10));
        EXPECT_EQ(st_list[0]["st_varchar"].get<std::string>(), "item_" + std::to_string(id));
        // nested Array leaf null round-trips as null
        ASSERT_TRUE(st_list[0]["st_tags"].is_array());
        EXPECT_EQ(st_list[0]["st_tags"][0].get<std::string>(), "a");
        EXPECT_TRUE(st_list[0]["st_tags"][1].is_null());
        // element-null sub-fields round-trip as null
        EXPECT_EQ(st_list[1]["st_int32"].get<int32_t>(), static_cast<int32_t>(id * 10 + 1));
        EXPECT_TRUE(st_list[1]["st_varchar"].is_null());
        EXPECT_TRUE(st_list[1]["st_vec"].is_null());
    }
}

TEST_F(MilvusServerTestStructElementNull, InsertAndQueryNullStructRow) {
    // a whole null struct row, an empty struct array, and a normal struct row round-trip:
    // the null row renders as JSON null, distinct from the empty array
    milvus::EntityRows rows_data;
    {
        milvus::EntityRow row;
        row["id"] = 0;
        row["st"] = nullptr;
        rows_data.emplace_back(std::move(row));
    }
    {
        milvus::EntityRow row;
        row["id"] = 1;
        row["st"] = nlohmann::json::array();
        rows_data.emplace_back(std::move(row));
    }
    {
        milvus::EntityRow row;
        row["id"] = 2;
        std::vector<nlohmann::json> struct_list;
        nlohmann::json st;
        st["st_int32"] = 10;
        st["st_varchar"] = "a";
        st["st_vec"] = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f};
        st["st_tags"] = nlohmann::json::array({"x"});
        struct_list.emplace_back(std::move(st));
        row["st"] = struct_list;
        rows_data.emplace_back(std::move(row));
    }

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 3);

    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("id >= 0");
    query_req.AddOutputField("id");
    query_req.AddOutputField("st");
    query_req.WithLimit(100);
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    status = query_resp.Results().OutputRows(rows);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(rows.size(), 3u);

    for (const auto& row : rows) {
        auto id = row["id"].get<int64_t>();
        if (id == 0) {
            EXPECT_TRUE(row["st"].is_null());
        } else if (id == 1) {
            ASSERT_TRUE(row["st"].is_array());
            EXPECT_TRUE(row["st"].empty());
        } else {
            ASSERT_TRUE(row["st"].is_array());
            ASSERT_EQ(row["st"].size(), 1u);
            EXPECT_EQ(row["st"][0]["st_int32"].get<int32_t>(), 10);
        }
    }
}

TEST_F(MilvusServerTestStructElementNull, AddAndDropStructSubField) {
    // add a struct sub-field by bracket path
    milvus::FieldSchema note("st[note]", milvus::DataType::VARCHAR);
    note.WithMaxLength(64).WithNullable(true);
    auto status = client_->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithCollectionName(collection_name).WithField(std::move(note)));
    milvus::test::ExpectStatusOK(status);

    // describe reflects the new sub-field with its plain name (the bracket prefix is
    // stripped, matching pymilvus)
    milvus::DescribeCollectionResponse desc_resp;
    status =
        client_->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name), desc_resp);
    milvus::test::ExpectStatusOK(status);
    const auto& struct_fields = desc_resp.Desc().Schema().StructFields();
    ASSERT_EQ(struct_fields.size(), 1u);
    const auto& st = struct_fields.at(0);
    EXPECT_TRUE(std::any_of(st.Fields().begin(), st.Fields().end(),
                            [](const milvus::FieldSchema& f) { return f.Name() == "note" && f.IsNullable(); }));

    // insert a row carrying the added sub-field with a plain key and read it back
    milvus::EntityRow row;
    row["id"] = 200;
    std::vector<milvus::EntityRow> struct_list;
    nlohmann::json st_row;
    st_row["st_int32"] = 1;
    st_row["note"] = "hello";
    struct_list.emplace_back(std::move(st_row));
    row["st"] = struct_list;
    milvus::InsertResponse insert_resp;
    status = client_->Insert(milvus::InsertRequest().WithCollectionName(collection_name).AddRowData(std::move(row)),
                             insert_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::QueryResponse query_resp;
    auto qrequest = milvus::QueryRequest()
                        .WithCollectionName(collection_name)
                        .WithFilter("id == 200")
                        .AddOutputField("id")
                        .AddOutputField("st")
                        .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    status = client_->Query(qrequest, query_resp);
    milvus::test::ExpectStatusOK(status);
    milvus::EntityRows rows;
    status = query_resp.Results().OutputRows(rows);
    milvus::test::ExpectStatusOK(status);
    ASSERT_EQ(rows.size(), 1u);
    ASSERT_EQ(rows[0]["st"].size(), 1u);
    EXPECT_EQ(rows[0]["st"][0]["note"].get<std::string>(), "hello");

    // drop the added sub-field
    status = client_->DropCollectionField(
        milvus::DropCollectionFieldRequest().WithCollectionName(collection_name).WithFieldName("st[note]"));
    milvus::test::ExpectStatusOK(status);

    status =
        client_->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name), desc_resp);
    milvus::test::ExpectStatusOK(status);
    const auto& st_after = desc_resp.Desc().Schema().StructFields().at(0);
    EXPECT_TRUE(std::none_of(st_after.Fields().begin(), st_after.Fields().end(),
                             [](const milvus::FieldSchema& f) { return f.Name() == "note"; }));
}
