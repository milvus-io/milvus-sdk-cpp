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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {
const std::string collection_name = "CPP_V2_PARTITION";
const std::string partition_name = "p1";
const std::string field_id = "id";
const std::string field_vector = "vector";
const uint32_t dimension = 4;

void
CreateCollection(milvus::MilvusClientV2Ptr& client) {
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, true});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));

    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    // a collection must have an index before its partitions can be loaded
    milvus::IndexParam index_vector(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndexParam(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);
}

void
CreatePartition(milvus::MilvusClientV2Ptr& client) {
    auto status = client->CreatePartition(
        milvus::CreatePartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name));
    util::CheckStatus("create partition: " + partition_name, status);
}

void
HasPartition(milvus::MilvusClientV2Ptr& client) {
    milvus::HasPartitionResponse response;
    auto status = client->HasPartition(
        milvus::HasPartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name), response);
    util::CheckStatus("check partition existence", status);
    std::cout << "Partition '" << partition_name << "' exists: " << (response.Has() ? "true" : "false") << std::endl;
}

void
LoadPartitions(milvus::MilvusClientV2Ptr& client) {
    auto status = client->LoadPartitions(
        milvus::LoadPartitionsRequest().WithCollectionName(collection_name).AddPartitionName(partition_name));
    util::CheckStatus("load partition: " + partition_name, status);
}

void
InsertIntoPartition(milvus::MilvusClientV2Ptr& client) {
    milvus::EntityRows rows;
    for (auto i = 0; i < 10; ++i) {
        milvus::EntityRow row;
        row[field_vector] = util::GenerateFloatVector(dimension);
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse response;
    auto status = client->Insert(milvus::InsertRequest()
                                     .WithCollectionName(collection_name)
                                     .WithPartitionName(partition_name)
                                     .WithRowsData(std::move(rows)),
                                 response);
    util::CheckStatus("insert rows into partition: " + partition_name, status);

    // flush to persist the data so the partition statistics are accurate
    status = client->Flush(milvus::FlushRequest().AddCollectionName(collection_name));
    util::CheckStatus("flush collection", status);
}

void
GetPartitionStats(milvus::MilvusClientV2Ptr& client) {
    milvus::GetPartitionStatsResponse response;
    auto status = client->GetPartitionStatistics(
        milvus::GetPartitionStatsRequest().WithCollectionName(collection_name).WithPartitionName(partition_name),
        response);
    util::CheckStatus("get partition stats", status);
    std::cout << "Partition '" << partition_name << "' stats: name=" << response.Stats().Name()
              << ", row_count=" << response.Stats().RowCount() << std::endl;
}

void
ReleasePartitions(milvus::MilvusClientV2Ptr& client) {
    auto status = client->ReleasePartitions(
        milvus::ReleasePartitionsRequest().WithCollectionName(collection_name).AddPartitionName(partition_name));
    util::CheckStatus("release partition: " + partition_name, status);
}

void
DropPartition(milvus::MilvusClientV2Ptr& client) {
    auto status = client->DropPartition(
        milvus::DropPartitionRequest().WithCollectionName(collection_name).WithPartitionName(partition_name));
    util::CheckStatus("drop partition: " + partition_name, status);
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    CreateCollection(client);
    CreatePartition(client);
    HasPartition(client);
    LoadPartitions(client);
    InsertIntoPartition(client);
    GetPartitionStats(client);
    ReleasePartitions(client);
    DropPartition(client);

    client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    client->Disconnect();
    return 0;
}
