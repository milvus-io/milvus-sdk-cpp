// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <map>
#include <string>
#include <vector>

#include "MilvusServerTest.h"

using milvus::test::MilvusServerTest;

namespace {

constexpr uint32_t dimension = 4;
constexpr int64_t max_capacity = 16;

const char* const field_id = "id";
const char* const field_vector = "vector";
const char* const field_bool = "array_bool";
const char* const field_int8 = "array_int8";
const char* const field_int16 = "array_int16";
const char* const field_int32 = "array_int32";
const char* const field_int64 = "array_int64";
const char* const field_float = "array_float";
const char* const field_double = "array_double";
const char* const field_varchar = "array_varchar";

void
AddArrayField(const milvus::CollectionSchemaPtr& schema, const std::string& name, milvus::DataType element_type) {
    auto field =
        milvus::FieldSchema(name, milvus::DataType::ARRAY).WithElementType(element_type).WithMaxCapacity(max_capacity);
    if (element_type == milvus::DataType::VARCHAR) {
        field.WithMaxLength(128);
    }
    schema->AddField(field);
}

template <typename Request>
void
AddArrayOutputFields(Request& request) {
    request.AddOutputField(field_id)
        .AddOutputField(field_bool)
        .AddOutputField(field_int8)
        .AddOutputField(field_int16)
        .AddOutputField(field_int32)
        .AddOutputField(field_int64)
        .AddOutputField(field_float)
        .AddOutputField(field_double)
        .AddOutputField(field_varchar);
}

milvus::EntityRow
CreateArrayRow(int64_t id, std::vector<float> vector, int64_t base) {
    return nlohmann::json{
        {field_id, id},
        {field_vector, std::move(vector)},
        {field_bool, {base % 2 == 0, base % 2 != 0}},
        {field_int8, {base + 1, base + 2}},
        {field_int16, {base + 10, base + 20}},
        {field_int32, {base + 100, base + 200}},
        {field_int64, {base + 1000, base + 2000}},
        {field_float, {base + 0.25, base + 0.5}},
        {field_double, {base + 0.125, base + 0.75}},
        {field_varchar, {"value_" + std::to_string(base), "value_" + std::to_string(base + 1)}},
    };
}

void
ExpectArrayFields(const milvus::EntityRow& actual, const milvus::EntityRow& expected) {
    EXPECT_EQ(actual.at(field_bool), expected.at(field_bool));
    EXPECT_EQ(actual.at(field_int8), expected.at(field_int8));
    EXPECT_EQ(actual.at(field_int16), expected.at(field_int16));
    EXPECT_EQ(actual.at(field_int32), expected.at(field_int32));
    EXPECT_EQ(actual.at(field_int64), expected.at(field_int64));
    EXPECT_EQ(actual.at(field_float), expected.at(field_float));
    EXPECT_EQ(actual.at(field_double), expected.at(field_double));
    EXPECT_EQ(actual.at(field_varchar), expected.at(field_varchar));
}

}  // namespace

class MilvusServerTestArray : public MilvusServerTest {
 protected:
    std::string collection_name;

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        collection_name = milvus::test::RanName("Array_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema(field_id, milvus::DataType::INT64, "id", true, false));
        schema->AddField(milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
        AddArrayField(schema, field_bool, milvus::DataType::BOOL);
        AddArrayField(schema, field_int8, milvus::DataType::INT8);
        AddArrayField(schema, field_int16, milvus::DataType::INT16);
        AddArrayField(schema, field_int32, milvus::DataType::INT32);
        AddArrayField(schema, field_int64, milvus::DataType::INT64);
        AddArrayField(schema, field_float, milvus::DataType::FLOAT);
        AddArrayField(schema, field_double, milvus::DataType::DOUBLE);
        AddArrayField(schema, field_varchar, milvus::DataType::VARCHAR);

        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        status = client_->CreateIndex(
            milvus::CreateIndexRequest()
                .WithCollectionName(collection_name)
                .AddIndex(milvus::IndexDesc(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::L2)));
        milvus::test::ExpectStatusOK(status);

        status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
        milvus::test::ExpectStatusOK(status);
    }

    void
    TearDown() override {
        client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
        MilvusServerTest::TearDown();
    }

    std::vector<int32_t>
    QueryInt32Array(int64_t id) {
        milvus::QueryResponse response;
        auto status = client_->Query(milvus::QueryRequest()
                                         .WithCollectionName(collection_name)
                                         .WithFilter(std::string(field_id) + " == " + std::to_string(id))
                                         .AddOutputField(field_int32)
                                         .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                                     response);
        milvus::test::ExpectStatusOK(status);

        milvus::EntityRows rows;
        status = response.Results().OutputRows(rows);
        milvus::test::ExpectStatusOK(status);
        EXPECT_EQ(rows.size(), 1);
        if (rows.size() != 1) {
            return {};
        }
        return rows[0].at(field_int32).get<std::vector<int32_t>>();
    }

    std::map<int64_t, milvus::EntityRow>
    InsertArrayRows() {
        milvus::EntityRows rows;
        rows.emplace_back(CreateArrayRow(1, {1.0f, 0.0f, 0.0f, 0.0f}, 0));
        rows.emplace_back(CreateArrayRow(2, {0.0f, 1.0f, 0.0f, 0.0f}, 10));

        std::map<int64_t, milvus::EntityRow> expected;
        for (const auto& row : rows) {
            expected.emplace(row.at(field_id).get<int64_t>(), row);
        }

        milvus::InsertResponse response;
        auto status = client_->Insert(
            milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), response);
        milvus::test::ExpectStatusOK(status);
        EXPECT_EQ(response.Results().InsertCount(), expected.size());
        return expected;
    }
};

TEST_F(MilvusServerTestArray, InsertAllElementTypes) {
    auto expected = InsertArrayRows();
    EXPECT_EQ(expected.size(), 2);
}

TEST_F(MilvusServerTestArray, QueryAllElementTypes) {
    const auto expected = InsertArrayRows();

    milvus::QueryRequest query_request;
    query_request.WithCollectionName(collection_name)
        .WithFilter("id > 0")
        .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    AddArrayOutputFields(query_request);

    milvus::QueryResponse query_response;
    auto status = client_->Query(query_request, query_response);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows query_rows;
    status = query_response.Results().OutputRows(query_rows);
    milvus::test::ExpectStatusOK(status);
    ASSERT_EQ(query_rows.size(), expected.size());
    for (const auto& row : query_rows) {
        const auto id = row.at(field_id).get<int64_t>();
        ASSERT_EQ(expected.count(id), 1);
        ExpectArrayFields(row, expected.at(id));
    }
}

TEST_F(MilvusServerTestArray, SearchAllElementTypes) {
    const auto expected = InsertArrayRows();

    milvus::SearchRequest search_request;
    search_request.WithCollectionName(collection_name)
        .WithAnnsField(field_vector)
        .WithLimit(2)
        .AddFloatVector({1.0f, 0.0f, 0.0f, 0.0f})
        .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    AddArrayOutputFields(search_request);

    milvus::SearchResponse search_response;
    auto status = client_->Search(search_request, search_response);
    milvus::test::ExpectStatusOK(status);

    const auto& results = search_response.Results().Results();
    ASSERT_EQ(results.size(), 1);
    milvus::EntityRows search_rows;
    status = results[0].OutputRows(search_rows);
    milvus::test::ExpectStatusOK(status);
    ASSERT_EQ(search_rows.size(), expected.size());
    for (const auto& row : search_rows) {
        const auto id = row.at(field_id).get<int64_t>();
        ASSERT_EQ(expected.count(id), 1);
        ExpectArrayFields(row, expected.at(id));
    }
}

TEST_F(MilvusServerTestArray, PartialUpsertArrayOperations) {
    milvus::EntityRows rows;
    rows.emplace_back(CreateArrayRow(10, {1.0f, 0.0f, 0.0f, 0.0f}, 0));

    milvus::InsertResponse insert_response;
    auto status = client_->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), insert_response);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_response.Results().InsertCount(), 1);

    auto upsert = [this](std::vector<int32_t> values, milvus::FieldPartialUpdateOp::OpType op_type,
                         bool partial_update) {
        milvus::EntityRow row;
        row[field_id] = 10;
        row[field_int32] = std::move(values);

        milvus::UpsertRequest request;
        request.WithCollectionName(collection_name)
            .WithPartialUpdate(partial_update)
            .AddRowData(std::move(row))
            .AddFieldOp(milvus::FieldPartialUpdateOp(field_int32, op_type));

        milvus::UpsertResponse response;
        auto status = client_->Upsert(request, response);
        milvus::test::ExpectStatusOK(status);
        EXPECT_EQ(response.Results().UpsertCount(), 1);
    };

    upsert({7, 8}, milvus::FieldPartialUpdateOp::OpType::REPLACE, true);
    EXPECT_EQ(QueryInt32Array(10), (std::vector<int32_t>{7, 8}));

    // ARRAY_APPEND concatenates the payload at the tail. Non-REPLACE operations
    // automatically enable partial update semantics in UpsertRequest.
    upsert({9, 8}, milvus::FieldPartialUpdateOp::OpType::ARRAY_APPEND, false);
    EXPECT_EQ(QueryInt32Array(10), (std::vector<int32_t>{7, 8, 9, 8}));

    // ARRAY_REMOVE removes every occurrence of each payload value.
    upsert({8}, milvus::FieldPartialUpdateOp::OpType::ARRAY_REMOVE, false);
    EXPECT_EQ(QueryInt32Array(10), (std::vector<int32_t>{7, 9}));
}
