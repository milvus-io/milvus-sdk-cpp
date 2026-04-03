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

        // primary key, auto_id=false
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, false));

        // scalar fields covering all scalar types
        schema->AddField(milvus::FieldSchema("f_bool", milvus::DataType::BOOL, "bool field"));
        schema->AddField(milvus::FieldSchema("f_int8", milvus::DataType::INT8, "int8 field"));
        schema->AddField(milvus::FieldSchema("f_int16", milvus::DataType::INT16, "int16 field"));
        schema->AddField(milvus::FieldSchema("f_int32", milvus::DataType::INT32, "int32 field"));
        schema->AddField(milvus::FieldSchema("f_int64", milvus::DataType::INT64, "int64 field"));
        schema->AddField(milvus::FieldSchema("f_float", milvus::DataType::FLOAT, "float field"));
        schema->AddField(milvus::FieldSchema("f_double", milvus::DataType::DOUBLE, "double field"));
        schema->AddField(
            milvus::FieldSchema("f_varchar", milvus::DataType::VARCHAR, "varchar field").WithMaxLength(128));
        schema->AddField(milvus::FieldSchema("f_json", milvus::DataType::JSON, "json field"));
        schema->AddField(milvus::FieldSchema("f_array", milvus::DataType::ARRAY, "array field")
                             .WithElementType(milvus::DataType::INT32)
                             .WithMaxCapacity(10));
        schema->AddField(milvus::FieldSchema("f_geo", milvus::DataType::GEOMETRY, "geometry field"));
        schema->AddField(milvus::FieldSchema("f_tsz", milvus::DataType::TIMESTAMPTZ, "timestamptz field"));

        // vector fields (max 4 per collection)
        schema->AddField(
            milvus::FieldSchema("v_float", milvus::DataType::FLOAT_VECTOR, "float vector").WithDimension(4));
        schema->AddField(
            milvus::FieldSchema("v_binary", milvus::DataType::BINARY_VECTOR, "binary vector").WithDimension(32));
        schema->AddField(
            milvus::FieldSchema("v_fp16", milvus::DataType::FLOAT16_VECTOR, "float16 vector").WithDimension(4));
        schema->AddField(milvus::FieldSchema("v_sparse", milvus::DataType::SPARSE_FLOAT_VECTOR, "sparse vector"));

        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        // create indexes for all vector fields
        status = client_->CreateIndex(
            milvus::CreateIndexRequest()
                .WithCollectionName(collection_name)
                .AddIndex(milvus::IndexDesc("v_float", "", milvus::IndexType::FLAT, milvus::MetricType::L2)));
        milvus::test::ExpectStatusOK(status);
        status = client_->CreateIndex(
            milvus::CreateIndexRequest()
                .WithCollectionName(collection_name)
                .AddIndex(milvus::IndexDesc("v_binary", "", milvus::IndexType::BIN_FLAT, milvus::MetricType::HAMMING)));
        milvus::test::ExpectStatusOK(status);
        status = client_->CreateIndex(
            milvus::CreateIndexRequest()
                .WithCollectionName(collection_name)
                .AddIndex(milvus::IndexDesc("v_fp16", "", milvus::IndexType::FLAT, milvus::MetricType::L2)));
        milvus::test::ExpectStatusOK(status);
        milvus::IndexDesc sparse_idx("v_sparse", "", milvus::IndexType::SPARSE_INVERTED_INDEX, milvus::MetricType::IP);
        sparse_idx.AddExtraParam("drop_ratio_build", "0.2");
        status = client_->CreateIndex(
            milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(sparse_idx)));
        milvus::test::ExpectStatusOK(status);
    }

    void
    TearDown() override {
        client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
        MilvusServerTest::TearDown();
    }
};

TEST_F(MilvusServerTestDml, InsertAndQuery) {
    // define test data once, used for both insert and verification
    milvus::EntityRows rows_data;
    rows_data.push_back(nlohmann::json{
        {"id", 1},
        {"f_bool", true},
        {"f_int8", 1},
        {"f_int16", 100},
        {"f_int32", 10000},
        {"f_int64", 100000},
        {"f_float", 1.5f},
        {"f_double", 2.5},
        {"f_varchar", "Alice"},
        {"f_json", nlohmann::json{{"key", "value1"}}},
        {"f_array", {10, 20, 30}},
        {"f_geo", "POINT (1 1)"},
        {"f_tsz", "2025-01-01T00:00:00+00:00"},
        {"v_float", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"v_binary", {255, 0, 171, 205}},
        {"v_fp16", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"v_sparse", nlohmann::json{{"1", 0.5}, {"3", 0.8}}},
    });
    rows_data.push_back(nlohmann::json{
        {"id", 2},
        {"f_bool", false},
        {"f_int8", 2},
        {"f_int16", 200},
        {"f_int32", 20000},
        {"f_int64", 200000},
        {"f_float", 3.5f},
        {"f_double", 4.5},
        {"f_varchar", "Bob"},
        {"f_json", nlohmann::json{{"key", "value2"}}},
        {"f_array", {40, 50}},
        {"f_geo", "POINT (2 2)"},
        {"f_tsz", "2025-06-15T12:30:00+08:00"},
        {"v_float", {0.5f, 0.6f, 0.7f, 0.8f}},
        {"v_binary", {128, 64, 32, 16}},
        {"v_fp16", {0.5f, 0.6f, 0.7f, 0.8f}},
        {"v_sparse", nlohmann::json{{"2", 0.3}, {"5", 0.9}}},
    });
    rows_data.push_back(nlohmann::json{
        {"id", 3},
        {"f_bool", true},
        {"f_int8", 3},
        {"f_int16", 300},
        {"f_int32", 30000},
        {"f_int64", 300000},
        {"f_float", 5.5f},
        {"f_double", 6.5},
        {"f_varchar", "Charlie"},
        {"f_json", nlohmann::json{{"key", "value3"}}},
        {"f_array", {60, 70, 80, 90}},
        {"f_geo", "LINESTRING (0 0, 1 1, 2 2)"},
        {"f_tsz", "2025-12-31T23:59:59-05:00"},
        {"v_float", {0.9f, 1.0f, 1.1f, 1.2f}},
        {"v_binary", {1, 2, 3, 4}},
        {"v_fp16", {0.9f, 1.0f, 1.1f, 1.2f}},
        {"v_sparse", nlohmann::json{{"0", 1.0}, {"4", 0.2}, {"7", 0.6}}},
    });

    // index by id for lookup during verification
    std::map<int64_t, nlohmann::json> expected;
    for (const auto& row : rows_data) {
        expected[row["id"].get<int64_t>()] = row;
    }

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 3);

    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    // query all rows
    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("id > 0");
    query_req.AddOutputField("f_bool");
    query_req.AddOutputField("f_int8");
    query_req.AddOutputField("f_int16");
    query_req.AddOutputField("f_int32");
    query_req.AddOutputField("f_int64");
    query_req.AddOutputField("f_float");
    query_req.AddOutputField("f_double");
    query_req.AddOutputField("f_varchar");
    query_req.AddOutputField("f_json");
    query_req.AddOutputField("f_array");
    query_req.AddOutputField("f_geo");
    query_req.AddOutputField("f_tsz");
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    query_resp.Results().OutputRows(rows);
    EXPECT_EQ(rows.size(), 3);

    // verify each row against the original inserted data
    for (const auto& row : rows) {
        auto key = row["id"].get<int64_t>();
        ASSERT_TRUE(expected.count(key)) << "unexpected id value: " << key;
        const auto& exp = expected[key];

        EXPECT_EQ(row["f_bool"].get<bool>(), exp["f_bool"].get<bool>());
        EXPECT_EQ(row["f_int8"].get<int8_t>(), exp["f_int8"].get<int8_t>());
        EXPECT_EQ(row["f_int16"].get<int16_t>(), exp["f_int16"].get<int16_t>());
        EXPECT_EQ(row["f_int32"].get<int32_t>(), exp["f_int32"].get<int32_t>());
        EXPECT_EQ(row["f_int64"].get<int64_t>(), exp["f_int64"].get<int64_t>());
        EXPECT_FLOAT_EQ(row["f_float"].get<float>(), exp["f_float"].get<float>());
        EXPECT_DOUBLE_EQ(row["f_double"].get<double>(), exp["f_double"].get<double>());
        EXPECT_EQ(row["f_varchar"].get<std::string>(), exp["f_varchar"].get<std::string>());
        EXPECT_EQ(row["f_json"], exp["f_json"]);
        EXPECT_EQ(row["f_array"].get<std::vector<int32_t>>(), exp["f_array"].get<std::vector<int32_t>>());
        EXPECT_EQ(row["f_geo"].get<std::string>(), exp["f_geo"].get<std::string>());
        EXPECT_TRUE(row.contains("f_tsz"));
    }
}

TEST_F(MilvusServerTestDml, DeleteByFilter) {
    milvus::EntityRows rows_data;
    rows_data.push_back(nlohmann::json{
        {"id", 1},
        {"f_bool", true},
        {"f_int8", 1},
        {"f_int16", 100},
        {"f_int32", 10000},
        {"f_int64", 100000},
        {"f_float", 1.5f},
        {"f_double", 2.5},
        {"f_varchar", "Alice"},
        {"f_json", nlohmann::json{{"k", "v1"}}},
        {"f_array", {1, 2}},
        {"f_geo", "POINT (1 1)"},
        {"f_tsz", "2025-01-01T00:00:00+00:00"},
        {"v_float", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"v_binary", {255, 0, 171, 205}},
        {"v_fp16", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"v_sparse", nlohmann::json{{"1", 0.5}}},
    });
    rows_data.push_back(nlohmann::json{
        {"id", 2},
        {"f_bool", false},
        {"f_int8", 2},
        {"f_int16", 200},
        {"f_int32", 20000},
        {"f_int64", 200000},
        {"f_float", 3.5f},
        {"f_double", 4.5},
        {"f_varchar", "Bob"},
        {"f_json", nlohmann::json{{"k", "v2"}}},
        {"f_array", {3, 4}},
        {"f_geo", "POINT (2 2)"},
        {"f_tsz", "2025-06-15T12:30:00+08:00"},
        {"v_float", {0.5f, 0.6f, 0.7f, 0.8f}},
        {"v_binary", {128, 64, 32, 16}},
        {"v_fp16", {0.5f, 0.6f, 0.7f, 0.8f}},
        {"v_sparse", nlohmann::json{{"2", 0.3}}},
    });

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);

    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    milvus::DeleteRequest delete_req;
    delete_req.WithCollectionName(collection_name).WithFilter("id == 1");
    milvus::DeleteResponse delete_resp;
    status = client_->Delete(delete_req, delete_resp);
    milvus::test::ExpectStatusOK(status);

    // query to verify deletion
    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("id == 1");
    query_req.AddOutputField("f_varchar");
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    query_resp.Results().OutputRows(rows);
    EXPECT_EQ(rows.size(), 0);
}

TEST_F(MilvusServerTestDml, Upsert) {
    milvus::EntityRows rows_data;
    rows_data.push_back(nlohmann::json{
        {"id", 1},
        {"f_bool", true},
        {"f_int8", 1},
        {"f_int16", 100},
        {"f_int32", 10000},
        {"f_int64", 100000},
        {"f_float", 1.5f},
        {"f_double", 2.5},
        {"f_varchar", "Alice"},
        {"f_json", nlohmann::json{{"k", "v1"}}},
        {"f_array", {1, 2}},
        {"f_geo", "POINT (1 1)"},
        {"f_tsz", "2025-01-01T00:00:00+00:00"},
        {"v_float", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"v_binary", {255, 0, 171, 205}},
        {"v_fp16", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"v_sparse", nlohmann::json{{"1", 0.5}}},
    });
    rows_data.push_back(nlohmann::json{
        {"id", 2},
        {"f_bool", false},
        {"f_int8", 2},
        {"f_int16", 200},
        {"f_int32", 20000},
        {"f_int64", 200000},
        {"f_float", 3.5f},
        {"f_double", 4.5},
        {"f_varchar", "Bob"},
        {"f_json", nlohmann::json{{"k", "v2"}}},
        {"f_array", {3, 4}},
        {"f_geo", "POINT (2 2)"},
        {"f_tsz", "2025-06-15T12:30:00+08:00"},
        {"v_float", {0.5f, 0.6f, 0.7f, 0.8f}},
        {"v_binary", {128, 64, 32, 16}},
        {"v_fp16", {0.5f, 0.6f, 0.7f, 0.8f}},
        {"v_sparse", nlohmann::json{{"2", 0.3}}},
    });

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);

    // upsert id=1 with updated data
    milvus::EntityRows upsert_rows;
    upsert_rows.push_back(nlohmann::json{
        {"id", 1},
        {"f_bool", false},
        {"f_int8", 11},
        {"f_int16", 101},
        {"f_int32", 10001},
        {"f_int64", 100001},
        {"f_float", 1.6f},
        {"f_double", 2.6},
        {"f_varchar", "Alice_v2"},
        {"f_json", nlohmann::json{{"k", "v1_updated"}}},
        {"f_array", {10, 20}},
        {"f_geo", "POINT (3 3)"},
        {"f_tsz", "2025-02-01T00:00:00+00:00"},
        {"v_float", {0.2f, 0.3f, 0.4f, 0.5f}},
        {"v_binary", {1, 2, 3, 4}},
        {"v_fp16", {0.2f, 0.3f, 0.4f, 0.5f}},
        {"v_sparse", nlohmann::json{{"1", 0.9}}},
    });

    milvus::UpsertRequest upsert_req;
    upsert_req.WithCollectionName(collection_name).WithRowsData(std::move(upsert_rows));
    milvus::UpsertResponse upsert_resp;
    status = client_->Upsert(upsert_req, upsert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(upsert_resp.Results().UpsertCount(), 1);
}

TEST_F(MilvusServerTestDml, ListPersistentSegments) {
    milvus::EntityRows rows_data;
    rows_data.push_back(nlohmann::json{
        {"id", 1},
        {"f_bool", true},
        {"f_int8", 1},
        {"f_int16", 100},
        {"f_int32", 10000},
        {"f_int64", 100000},
        {"f_float", 1.5f},
        {"f_double", 2.5},
        {"f_varchar", "Alice"},
        {"f_json", nlohmann::json{{"k", "v"}}},
        {"f_array", {1}},
        {"f_geo", "POINT (1 1)"},
        {"f_tsz", "2025-01-01T00:00:00+00:00"},
        {"v_float", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"v_binary", {255, 0, 171, 205}},
        {"v_fp16", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"v_sparse", nlohmann::json{{"1", 0.5}}},
    });

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithRowsData(std::move(rows_data));
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
}
