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
