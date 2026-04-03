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

class MilvusServerTestPartition : public MilvusServerTest {
 protected:
    std::string collection_name;

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        collection_name = milvus::test::RanName("PartTest_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        schema->AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(4));

        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        milvus::IndexDesc index_desc("vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
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

TEST_F(MilvusServerTestPartition, CreateHasListDrop) {
    std::string partition_name = milvus::test::RanName("Part_");

    // create partition
    auto status = client_->CreatePartition(
        milvus::CreatePartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name));
    milvus::test::ExpectStatusOK(status);

    // has partition
    milvus::HasPartitionResponse has_resp;
    status = client_->HasPartition(
        milvus::HasPartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name), has_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_TRUE(has_resp.Has());

    // list partitions - should have _default and the new one
    milvus::ListPartitionsResponse list_resp;
    status = client_->ListPartitions(milvus::ListPartitionsRequest().WithCollectionName(collection_name), list_resp);
    milvus::test::ExpectStatusOK(status);
    auto names = list_resp.PartitionsNames();
    EXPECT_GE(names.size(), 2);
    EXPECT_NE(std::find(names.begin(), names.end(), partition_name), names.end());

    // drop partition
    status = client_->DropPartition(
        milvus::DropPartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name));
    milvus::test::ExpectStatusOK(status);

    // verify dropped
    milvus::HasPartitionResponse has_resp2;
    status = client_->HasPartition(
        milvus::HasPartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name), has_resp2);
    milvus::test::ExpectStatusOK(status);
    EXPECT_FALSE(has_resp2.Has());

    // list partitions again - the dropped one should not be in the list, but _default should still be there
    status = client_->ListPartitions(milvus::ListPartitionsRequest().WithCollectionName(collection_name), list_resp);
    milvus::test::ExpectStatusOK(status);
    names = list_resp.PartitionsNames();
    EXPECT_GE(names.size(), 1);
    EXPECT_EQ(std::find(names.begin(), names.end(), partition_name), names.end());
}

TEST_F(MilvusServerTestPartition, LoadAndReleasePartitions) {
    std::string partition_name = milvus::test::RanName("Part_");

    auto status = client_->CreatePartition(
        milvus::CreatePartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name));
    milvus::test::ExpectStatusOK(status);

    // load partitions
    status = client_->LoadPartitions(
        milvus::LoadPartitionsRequest().WithCollectionName(collection_name).AddPartitionName(partition_name));
    milvus::test::ExpectStatusOK(status);

    // check load state of the partition, should be loaded
    auto get_load_state_req =
        milvus::GetLoadStateRequest().WithCollectionName(collection_name).AddPartitionName(partition_name);
    milvus::GetLoadStateResponse load_state_resp;
    status = client_->GetLoadState(get_load_state_req, load_state_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(milvus::LoadState::LOAD_STATE_LOADED, load_state_resp.State());

    // release partitions
    status = client_->ReleasePartitions(
        milvus::ReleasePartitionsRequest().WithCollectionName(collection_name).AddPartitionName(partition_name));
    milvus::test::ExpectStatusOK(status);

    // check load state of the collection, should be not load since the only partition is released
    status = client_->GetLoadState(get_load_state_req, load_state_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(milvus::LoadState::LOAD_STATE_NOT_LOAD, load_state_resp.State());
}

TEST_F(MilvusServerTestPartition, GetPartitionStatistics) {
    std::string partition_name = milvus::test::RanName("Part_");

    auto status = client_->CreatePartition(
        milvus::CreatePartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name));
    milvus::test::ExpectStatusOK(status);

    // insert data into the partition
    std::vector<milvus::FieldDataPtr> fields{std::make_shared<milvus::FloatVecFieldData>(
        "vec", std::vector<std::vector<float>>{{0.1f, 0.2f, 0.3f, 0.4f}, {0.5f, 0.6f, 0.7f, 0.8f}})};

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithPartitionName(partition_name);
    insert_req.WithColumnsData(std::move(fields));
    milvus::InsertResponse insert_resp;
    status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);

    // flush to ensure data is persisted
    status = client_->Flush(milvus::FlushRequest().AddCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    // get partition statistics
    milvus::GetPartitionStatsResponse stats_resp;
    status = client_->GetPartitionStatistics(
        milvus::GetPartitionStatsRequest().WithCollectionName(collection_name).WithPartitionName(partition_name),
        stats_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(stats_resp.Stats().RowCount(), 2);
}

TEST_F(MilvusServerTestPartition, PartitionKey) {
    // create a separate collection with partition key field
    std::string pk_coll = milvus::test::RanName("PKTest_");

    auto schema = std::make_shared<milvus::CollectionSchema>(pk_coll);
    schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
    schema->AddField(milvus::FieldSchema("category", milvus::DataType::VARCHAR, "partition key")
                         .WithMaxLength(64)
                         .WithPartitionKey(true));
    schema->AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(4));

    auto status = client_->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(pk_coll).WithCollectionSchema(schema).WithNumPartitions(
            4));
    milvus::test::ExpectStatusOK(status);

    // verify partitions are auto-created (should have more than 1 partition)
    milvus::ListPartitionsResponse lp_resp;
    status = client_->ListPartitions(milvus::ListPartitionsRequest().WithCollectionName(pk_coll), lp_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_GE(lp_resp.PartitionsNames().size(), 4);

    // create index and load
    milvus::IndexDesc idx("vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    status = client_->CreateIndex(milvus::CreateIndexRequest().WithCollectionName(pk_coll).AddIndex(std::move(idx)));
    milvus::test::ExpectStatusOK(status);

    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(pk_coll));
    milvus::test::ExpectStatusOK(status);

    // insert data with different category values — milvus distributes them across partitions
    milvus::EntityRows rows_data;
    std::vector<std::string> categories = {"electronics", "books", "clothing", "food"};
    for (int i = 0; i < 40; ++i) {
        nlohmann::json row;
        row["category"] = categories[i % 4];
        row["vec"] = std::vector<float>{0.1f * (i + 1), 0.2f, 0.3f, 0.4f};
        rows_data.emplace_back(std::move(row));
    }

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(pk_coll).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 40);

    // query with partition key filter — milvus only scans the relevant partition
    milvus::QueryRequest query_req;
    query_req.WithCollectionName(pk_coll);
    query_req.WithFilter(R"(category == "books")");
    query_req.AddOutputField("category");
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    query_resp.Results().OutputRows(rows);
    EXPECT_EQ(rows.size(), 10);  // 40 rows / 4 categories = 10 per category
    for (const auto& row : rows) {
        EXPECT_EQ(row["category"].get<std::string>(), "books");
    }

    // search with partition key filter
    milvus::SearchRequest search_req;
    search_req.WithCollectionName(pk_coll);
    search_req.WithAnnsField("vec");
    search_req.WithLimit(5);
    search_req.AddFloatVector({0.5f, 0.2f, 0.3f, 0.4f});
    search_req.WithFilter(R"(category == "electronics")");
    search_req.AddOutputField("category");
    search_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::SearchResponse search_resp;
    status = client_->Search(search_req, search_resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = search_resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results.at(0).Scores().size(), 5);

    milvus::EntityRows search_rows;
    results.at(0).OutputRows(search_rows);
    for (const auto& row : search_rows) {
        EXPECT_EQ(row["category"].get<std::string>(), "electronics");
    }

    // cleanup
    client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(pk_coll));
}
