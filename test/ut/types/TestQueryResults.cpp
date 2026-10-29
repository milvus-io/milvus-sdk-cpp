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

#include <gtest/gtest.h>

#include "milvus/types/Iterator.h"
#include "milvus/types/QueryResults.h"

class QueryResultsTest : public ::testing::Test {};

TEST_F(QueryResultsTest, GeneralTesting) {
    std::vector<milvus::FieldDataPtr> fields{
        nullptr,
        std::make_shared<milvus::BoolFieldData>("bool_data"),
        std::make_shared<milvus::Int16FieldData>("int16_data"),
    };
    std::set<std::string> output_names;
    output_names.insert("int16_data");

    milvus::QueryResults results(std::move(fields), output_names);
    EXPECT_EQ(results.GetFieldByName("foo"), nullptr);
    EXPECT_EQ(results.GetFieldByName("int16_data")->Name(), "int16_data");
    EXPECT_EQ(results.OutputFields().front(), nullptr);
    EXPECT_EQ(results.OutputFieldNames().size(), output_names.size());
}

TEST_F(QueryResultsTest, OutputRowsAndOutputRow) {
    // Create fields with actual data
    auto int_field = std::make_shared<milvus::Int64FieldData>("id");
    int_field->Add(100);
    int_field->Add(200);

    auto str_field = std::make_shared<milvus::VarCharFieldData>("name");
    str_field->Add("alice");
    str_field->Add("bob");

    std::vector<milvus::FieldDataPtr> fields{int_field, str_field};
    std::set<std::string> output_names{"id", "name"};

    milvus::QueryResults results(std::move(fields), output_names);

    // Test OutputRows
    milvus::EntityRows rows;
    auto status = results.OutputRows(rows);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rows.size(), 2);
    EXPECT_EQ(results.GetRowCount(), 2);

    // Test OutputRow
    milvus::EntityRow row;
    status = results.OutputRow(0, row);
    EXPECT_TRUE(status.IsOk());

    status = results.OutputRow(1, row);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(QueryResultsTest, Clear) {
    auto int_field = std::make_shared<milvus::Int64FieldData>("id");
    int_field->Add(100);

    std::vector<milvus::FieldDataPtr> fields{int_field};
    std::set<std::string> output_names{"id"};

    milvus::QueryResults results(std::move(fields), output_names);
    EXPECT_EQ(results.OutputFields().size(), 1);

    results.Clear();
    EXPECT_EQ(results.OutputFields().size(), 0);
}

TEST_F(QueryResultsTest, ElementIndices) {
    std::vector<milvus::FieldDataPtr> fields;
    std::set<std::string> output_names;
    milvus::QueryResults results(std::move(fields), output_names);
    EXPECT_TRUE(results.ElementIndices().empty());

    std::vector<std::vector<int64_t>> element_indices{{2, 5}, {9}};
    results.SetElementIndices(std::move(element_indices));
    ASSERT_EQ(results.ElementIndices().size(), 2);
    EXPECT_EQ(results.ElementIndices()[0], (std::vector<int64_t>{2, 5}));
    EXPECT_EQ(results.ElementIndices()[1], (std::vector<int64_t>{9}));

    results.Clear();
    EXPECT_TRUE(results.ElementIndices().empty());
}

TEST_F(QueryResultsTest, QueryIteratorCursor) {
    milvus::QueryIteratorCursor cursor;
    EXPECT_FALSE(cursor.HasIntPk());
    EXPECT_FALSE(cursor.HasStrPk());
    EXPECT_FALSE(cursor.HasLastElementOffset());

    EXPECT_TRUE(cursor.SetSessionTs(12345).IsOk());
    EXPECT_EQ(cursor.SessionTs(), 12345);
    cursor.WithSessionTs(54321);
    EXPECT_EQ(cursor.SessionTs(), 54321);

    EXPECT_TRUE(cursor.SetIntPk(99).IsOk());
    EXPECT_TRUE(cursor.HasIntPk());
    EXPECT_FALSE(cursor.HasStrPk());
    EXPECT_EQ(cursor.IntPk(), 99);

    EXPECT_TRUE(cursor.SetStrPk("row-7").IsOk());
    EXPECT_TRUE(cursor.HasStrPk());
    EXPECT_FALSE(cursor.HasIntPk());
    EXPECT_EQ(cursor.StrPk(), "row-7");
    cursor.WithStrPk("row-8");
    EXPECT_EQ(cursor.StrPk(), "row-8");

    EXPECT_TRUE(cursor.SetLastElementOffset(3).IsOk());
    EXPECT_TRUE(cursor.HasLastElementOffset());
    EXPECT_EQ(cursor.LastElementOffset(), 3);
    cursor.WithLastElementOffset(4);
    EXPECT_EQ(cursor.LastElementOffset(), 4);

    auto negative = cursor.SetLastElementOffset(-1);
    EXPECT_FALSE(negative.IsOk());
    EXPECT_EQ(negative.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_TRUE(cursor.HasLastElementOffset());
}
