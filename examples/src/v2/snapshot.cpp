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

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {

const char* const collection_name = "CPP_V2_SNAPSHOT";
const char* const restore_collection_name = "CPP_V2_SNAPSHOT_RESTORE";
const char* const snapshot_name = "cpp_sdk_example_snapshot_backup";
const char* const field_id = "id";
const char* const field_vector = "vector";
const uint32_t dimension = 4;
const int64_t row_count = 1000;

uint64_t
queryRowCount(milvus::MilvusClientV2Ptr& client, const std::string& name) {
    auto request = milvus::QueryRequest()
                       .WithCollectionName(name)
                       .AddOutputField("count(*)")
                       .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse response;
    auto status = client->Query(request, response);
    util::CheckStatus("query count(*) on collection: " + name, status);
    return response.Results().GetRowCount();
}

void
printRowCount(milvus::MilvusClientV2Ptr& client, const std::string& name) {
    std::cout << name << " count(*) = " << queryRowCount(client, name) << std::endl;
}

uint64_t
getPersistedRowCount(milvus::MilvusClientV2Ptr& client, const std::string& name) {
    milvus::GetCollectionStatsResponse response;
    auto status = client->GetCollectionStats(milvus::GetCollectionStatsRequest().WithCollectionName(name), response);
    util::CheckStatus("get collection stats: " + name, status);
    return response.Stats().RowCount();
}

milvus::RestoreSnapshotJobInfo
waitRestoreSnapshot(milvus::MilvusClientV2Ptr& client, int64_t job_id) {
    for (int i = 0; i < 60; ++i) {
        milvus::GetRestoreSnapshotStateResponse response;
        auto status =
            client->GetRestoreSnapshotState(milvus::GetRestoreSnapshotStateRequest().WithJobID(job_id), response);
        util::CheckStatus("get restore snapshot state", status);

        const auto& job_info = response.JobInfo();
        if (job_info.State() == milvus::RestoreSnapshotStateCode::COMPLETED) {
            return job_info;
        }
        if (job_info.State() == milvus::RestoreSnapshotStateCode::FAILED) {
            std::cerr << "Restore snapshot failed: " << job_info.Reason() << std::endl;
            exit(1);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cerr << "Restore snapshot did not complete within 60 seconds" << std::endl;
    exit(1);
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"http://localhost:19530"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(restore_collection_name));
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));

    status = client->CreateCollection(milvus::CreateSimpleCollectionRequest()
                                          .WithCollectionName(collection_name)
                                          .WithPrimaryFieldName(field_id)
                                          .WithVectorFieldName(field_vector)
                                          .WithDimension(dimension));
    util::CheckStatus("create collection: " + std::string(collection_name), status);
    std::cout << "Collection '" << collection_name << "' created" << std::endl;

    milvus::EntityRows rows;
    rows.reserve(row_count);
    for (int64_t i = 0; i < row_count; ++i) {
        milvus::EntityRow row;
        row[field_id] = i;
        row[field_vector] = std::vector<float>{static_cast<float>(i), static_cast<float>(i) / 2,
                                               static_cast<float>(i) / 3, static_cast<float>(i) / 4};
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse insert_response;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            insert_response);
    util::CheckStatus("insert", status);
    std::cout << insert_response.Results().InsertCount() << " rows inserted" << std::endl;

    printRowCount(client, collection_name);

    status = client->Flush(milvus::FlushRequest().AddCollectionName(collection_name));
    util::CheckStatus("flush collection: " + std::string(collection_name), status);
    std::cout << "Collection flushed" << std::endl;

    status = client->CreateSnapshot(milvus::CreateSnapshotRequest()
                                        .WithCollectionName(collection_name)
                                        .WithSnapshotName(snapshot_name)
                                        .WithDescription("Snapshot example backup"));
    util::CheckStatus("create snapshot: " + std::string(snapshot_name), status);
    std::cout << "Snapshot '" << snapshot_name << "' created" << std::endl;

    milvus::ListSnapshotsResponse list_response;
    status = client->ListSnapshots(milvus::ListSnapshotsRequest().WithCollectionName(collection_name), list_response);
    util::CheckStatus("list snapshots", status);
    std::cout << "Snapshots:" << std::endl;
    for (const auto& name : list_response.Snapshots()) {
        std::cout << "\t" << name << std::endl;
    }

    milvus::DescribeSnapshotResponse describe_response;
    status = client->DescribeSnapshot(
        milvus::DescribeSnapshotRequest().WithCollectionName(collection_name).WithSnapshotName(snapshot_name),
        describe_response);
    util::CheckStatus("describe snapshot: " + std::string(snapshot_name), status);
    std::cout << "Snapshot detail: name=" << describe_response.Name()
              << ", collection=" << describe_response.CollectionName() << ", partitions=";
    for (size_t i = 0; i < describe_response.PartitionNames().size(); ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << describe_response.PartitionNames()[i];
    }
    std::cout << ", createTs=" << describe_response.CreateTs() << ", s3Location=" << describe_response.S3Location()
              << std::endl;

    milvus::PinSnapshotDataResponse pin_response;
    status = client->PinSnapshotData(milvus::PinSnapshotDataRequest()
                                         .WithCollectionName(collection_name)
                                         .WithSnapshotName(snapshot_name)
                                         .WithTtlSeconds(3600),
                                     pin_response);
    util::CheckStatus("pin snapshot data", status);
    std::cout << "Snapshot data pinned, pinId=" << pin_response.PinID() << std::endl;

    milvus::RestoreSnapshotResponse restore_response;
    status = client->RestoreSnapshot(milvus::RestoreSnapshotRequest()
                                         .WithSnapshotName(snapshot_name)
                                         .WithSourceCollectionName(collection_name)
                                         .WithTargetCollectionName(restore_collection_name),
                                     restore_response);
    util::CheckStatus("restore snapshot", status);
    std::cout << "Restore snapshot job submitted, jobId=" << restore_response.JobID() << std::endl;

    auto job_info = waitRestoreSnapshot(client, restore_response.JobID());
    std::cout << "Restore job state: state=" << std::to_string(job_info.State()) << ", progress=" << job_info.Progress()
              << ", reason=" << job_info.Reason() << std::endl;

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(restore_collection_name));
    util::CheckStatus("load restored collection: " + std::string(restore_collection_name), status);

    auto source_row_count = getPersistedRowCount(client, collection_name);
    auto target_row_count = getPersistedRowCount(client, restore_collection_name);
    std::cout << "Source persisted row count=" << source_row_count << ", target persisted row count="
              << target_row_count << std::endl;
    if (source_row_count != target_row_count) {
        std::cerr << "Restored row count mismatch: source=" << source_row_count << ", target=" << target_row_count
                  << std::endl;
        exit(1);
    }

    status = client->UnpinSnapshotData(milvus::UnpinSnapshotDataRequest().WithPinID(pin_response.PinID()));
    util::CheckStatus("unpin snapshot data", status);
    std::cout << "Snapshot data unpinned, pinId=" << pin_response.PinID() << std::endl;

    status = client->DropSnapshot(
        milvus::DropSnapshotRequest().WithCollectionName(collection_name).WithSnapshotName(snapshot_name));
    util::CheckStatus("drop snapshot: " + std::string(snapshot_name), status);
    std::cout << "Snapshot '" << snapshot_name << "' dropped" << std::endl;

    client->Disconnect();
    return 0;
}
