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

#include "milvus/MilvusClientV2.h"

class InsertRequestTest : public ::testing::Test {};

TEST_F(InsertRequestTest, GettersAndSetters) {
    milvus::InsertRequest req;

    req.WithCollectionName("insert_coll");
    EXPECT_EQ(req.CollectionName(), "insert_coll");

    req.WithPartitionName("insert_part");
    EXPECT_EQ(req.PartitionName(), "insert_part");

    // ColumnsData
    auto field_data = std::make_shared<milvus::Int64FieldData>("id");
    std::vector<milvus::FieldDataPtr> columns;
    columns.push_back(field_data);
    req.WithColumnsData(std::move(columns));
    EXPECT_EQ(req.ColumnsData().size(), 1);

    // AddColumnData
    auto field_data2 = std::make_shared<milvus::Int64FieldData>("id2");
    req.AddColumnData(field_data2);
    EXPECT_EQ(req.ColumnsData().size(), 2);

    // RowsData
    milvus::InsertRequest req2;
    milvus::EntityRow row;
    row["name"] = "test";
    milvus::EntityRows rows;
    rows.push_back(row);
    req2.WithRowsData(std::move(rows));
    EXPECT_EQ(req2.RowsData().size(), 1);

    // AddRowData
    milvus::EntityRow row2;
    row2["name"] = "test2";
    req2.AddRowData(std::move(row2));
    EXPECT_EQ(req2.RowsData().size(), 2);
}

class UpsertRequestTest : public ::testing::Test {};

TEST_F(UpsertRequestTest, GettersAndSetters) {
    milvus::UpsertRequest req;

    req.WithCollectionName("upsert_coll");
    EXPECT_EQ(req.CollectionName(), "upsert_coll");

    req.WithPartitionName("upsert_part");
    EXPECT_EQ(req.PartitionName(), "upsert_part");

    // ColumnsData
    auto field_data = std::make_shared<milvus::Int64FieldData>("id");
    std::vector<milvus::FieldDataPtr> columns;
    columns.push_back(field_data);
    req.WithColumnsData(std::move(columns));
    EXPECT_EQ(req.ColumnsData().size(), 1);

    // RowsData
    milvus::UpsertRequest req2;
    milvus::EntityRow row;
    row["name"] = "test";
    milvus::EntityRows rows;
    rows.push_back(row);
    req2.WithRowsData(std::move(rows));
    EXPECT_EQ(req2.RowsData().size(), 1);

    // PartialUpdate
    req.WithPartialUpdate(true);
    EXPECT_TRUE(req.PartialUpdate());
    req.WithPartialUpdate(false);
    EXPECT_FALSE(req.PartialUpdate());
}

class DeleteRequestTest : public ::testing::Test {};

TEST_F(DeleteRequestTest, GettersAndSetters) {
    milvus::DeleteRequest req;

    req.WithCollectionName("delete_coll");
    EXPECT_EQ(req.CollectionName(), "delete_coll");

    req.WithPartitionName("delete_part");
    EXPECT_EQ(req.PartitionName(), "delete_part");

    // Filter
    req.WithFilter("id > 100");
    EXPECT_EQ(req.Filter(), "id > 100");

    // Note: DeleteRequest::AddFilterTemplate has ambiguous overloads (std::string vs const std::string&),
    // skip testing it here. The method is tested via integration tests.

    // IDs (int64)
    milvus::DeleteRequest req2;
    std::vector<int64_t> int_ids{1, 2, 3};
    req2.WithIDs(std::move(int_ids));
    EXPECT_FALSE(req2.IDs().IntIDArray().empty());

    // IDs (string)
    milvus::DeleteRequest req3;
    std::vector<std::string> str_ids{"a", "b", "c"};
    req3.WithIDs(std::move(str_ids));
    EXPECT_FALSE(req3.IDs().StrIDArray().empty());
}

TEST_F(DeleteRequestTest, AllMethods) {
    milvus::DeleteRequest req;

    // DMLRequestBase methods
    EXPECT_TRUE(req.DatabaseName().empty());
    req.SetDatabaseName("db1");
    EXPECT_EQ(req.DatabaseName(), "db1");
    req.WithDatabaseName("db2");
    EXPECT_EQ(req.DatabaseName(), "db2");

    req.SetCollectionName("coll1");
    EXPECT_EQ(req.CollectionName(), "coll1");

    req.SetPartitionName("part1");
    EXPECT_EQ(req.PartitionName(), "part1");

    // SetFilter
    req.SetFilter("age > 10");
    EXPECT_EQ(req.Filter(), "age > 10");

    // SetFilterTemplates / WithFilterTemplates
    std::unordered_map<std::string, nlohmann::json> tmpls;
    tmpls["threshold"] = 100;
    req.SetFilterTemplates(std::move(tmpls));
    EXPECT_EQ(req.FilterTemplates().size(), 1);
    EXPECT_EQ(req.FilterTemplates().at("threshold"), 100);

    std::unordered_map<std::string, nlohmann::json> tmpls2;
    tmpls2["names"] = nlohmann::json{"a", "b"};
    tmpls2["age"] = 25;
    req.WithFilterTemplates(std::move(tmpls2));
    EXPECT_EQ(req.FilterTemplates().size(), 2);

    // SetIDs (int64)
    std::vector<int64_t> ids{10, 20, 30};
    req.SetIDs(std::move(ids));
    EXPECT_EQ(req.IDs().IntIDArray().size(), 3);

    // SetIDs (string)
    milvus::DeleteRequest req2;
    std::vector<std::string> str_ids{"x", "y"};
    req2.SetIDs(std::move(str_ids));
    EXPECT_EQ(req2.IDs().StrIDArray().size(), 2);
}

TEST_F(UpsertRequestTest, AllMethods) {
    milvus::UpsertRequest req;

    // DMLRequestBase methods via UpsertRequest overrides
    req.WithDatabaseName("db1");
    EXPECT_EQ(req.DatabaseName(), "db1");

    req.WithCollectionName("coll1");
    EXPECT_EQ(req.CollectionName(), "coll1");

    req.WithPartitionName("part1");
    EXPECT_EQ(req.PartitionName(), "part1");

    // SetPartialUpdate
    req.SetPartialUpdate(true);
    EXPECT_TRUE(req.PartialUpdate());
    req.SetPartialUpdate(false);
    EXPECT_FALSE(req.PartialUpdate());

    // AddColumnData
    auto field = std::make_shared<milvus::Int64FieldData>("id");
    req.AddColumnData(field);
    EXPECT_EQ(req.ColumnsData().size(), 1);

    // AddRowData
    milvus::UpsertRequest req2;
    milvus::EntityRow row;
    row["name"] = "test";
    req2.AddRowData(std::move(row));
    EXPECT_EQ(req2.RowsData().size(), 1);
}

TEST_F(InsertRequestTest, DMLRequestBaseMethods) {
    milvus::InsertRequest req;

    // DatabaseName
    EXPECT_TRUE(req.DatabaseName().empty());
    req.SetDatabaseName("db1");
    EXPECT_EQ(req.DatabaseName(), "db1");
    req.WithDatabaseName("db2");
    EXPECT_EQ(req.DatabaseName(), "db2");

    // SetCollectionName / SetPartitionName
    req.SetCollectionName("coll1");
    EXPECT_EQ(req.CollectionName(), "coll1");

    req.SetPartitionName("part1");
    EXPECT_EQ(req.PartitionName(), "part1");

    // SetColumnsData / SetRowsData
    std::vector<milvus::FieldDataPtr> cols;
    cols.push_back(std::make_shared<milvus::Int64FieldData>("id"));
    req.SetColumnsData(std::move(cols));
    EXPECT_EQ(req.ColumnsData().size(), 1);

    milvus::InsertRequest req2;
    milvus::EntityRows rows;
    rows.push_back(nlohmann::json{{"name", "test"}});
    req2.SetRowsData(std::move(rows));
    EXPECT_EQ(req2.RowsData().size(), 1);
}
