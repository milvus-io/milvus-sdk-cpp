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

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::CreateSnapshotRequest;
using ::milvus::proto::milvus::DescribeSnapshotRequest;
using ::milvus::proto::milvus::DescribeSnapshotResponse;
using ::milvus::proto::milvus::DropSnapshotRequest;
using ::milvus::proto::milvus::GetRestoreSnapshotStateRequest;
using ::milvus::proto::milvus::GetRestoreSnapshotStateResponse;
using ::milvus::proto::milvus::ListRestoreSnapshotJobsRequest;
using ::milvus::proto::milvus::ListRestoreSnapshotJobsResponse;
using ::milvus::proto::milvus::ListSnapshotsRequest;
using ::milvus::proto::milvus::ListSnapshotsResponse;
using ::milvus::proto::milvus::PinSnapshotDataRequest;
using ::milvus::proto::milvus::PinSnapshotDataResponse;
using ::milvus::proto::milvus::RestoreSnapshotInfo;
using ::milvus::proto::milvus::RestoreSnapshotRequest;
using ::milvus::proto::milvus::RestoreSnapshotResponse;
using ::milvus::proto::milvus::UnpinSnapshotDataRequest;
using ::testing::_;

namespace {

milvus::MilvusClientV2Ptr
CreateConnectedV2Client(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{"127.0.0.1", port};
    auto status = client->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());
    return client;
}

void
FillRestoreSnapshotInfo(RestoreSnapshotInfo* info) {
    info->set_job_id(101);
    info->set_snapshot_name("snap");
    info->set_db_name("db");
    info->set_collection_name("coll");
    info->set_state(::milvus::proto::milvus::RestoreSnapshotState::RestoreSnapshotCompleted);
    info->set_progress(100);
    info->set_reason("done");
    info->set_start_time(1000);
    info->set_time_cost(2000);
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, CreateSnapshot) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, CreateSnapshot(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const CreateSnapshotRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_EQ(request->name(), "snap");
            EXPECT_EQ(request->description(), "desc");
            EXPECT_EQ(request->compaction_protection_seconds(), 9);
            return ::grpc::Status{};
        });

    milvus::CreateSnapshotRequest request;
    request.WithDatabaseName("db")
        .WithCollectionName("coll")
        .WithSnapshotName("snap")
        .WithDescription("desc")
        .WithCompactionProtectionSeconds(9);

    auto status = client->CreateSnapshot(request);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, CreateSnapshotInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::CreateSnapshotRequest request;
    request.WithCollectionName("coll");
    auto status = client->CreateSnapshot(request);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, CreateSnapshotNegativeCompactionProtectionInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::CreateSnapshotRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll").WithSnapshotName("snap").WithCompactionProtectionSeconds(
        -1);

    auto status = client->CreateSnapshot(request);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, CreateSnapshotFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, CreateSnapshot(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const CreateSnapshotRequest*, ::milvus::proto::common::Status*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::CreateSnapshotRequest request;
    request.WithDatabaseName("db")
        .WithCollectionName("coll")
        .WithSnapshotName("snap")
        .WithDescription("desc")
        .WithCompactionProtectionSeconds(9);

    auto status = client->CreateSnapshot(request);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, DropSnapshot) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DropSnapshot(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const DropSnapshotRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_EQ(request->name(), "snap");
            return ::grpc::Status{};
        });

    milvus::DropSnapshotRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll").WithSnapshotName("snap");
    auto status = client->DropSnapshot(request);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, DropSnapshotInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::DropSnapshotRequest request;
    request.WithCollectionName("coll");
    auto status = client->DropSnapshot(request);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, DropSnapshotFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DropSnapshot(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const DropSnapshotRequest*, ::milvus::proto::common::Status*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::DropSnapshotRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll").WithSnapshotName("snap");
    auto status = client->DropSnapshot(request);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, ListSnapshots) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListSnapshots(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListSnapshotsRequest* request, ListSnapshotsResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collection_name(), "coll");
            response->add_snapshots("snap1");
            response->add_snapshots("snap2");
            return ::grpc::Status{};
        });

    milvus::ListSnapshotsRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll");
    milvus::ListSnapshotsResponse response;
    auto status = client->ListSnapshots(request, response);

    EXPECT_TRUE(status.IsOk());
    ASSERT_EQ(response.Snapshots().size(), 2);
    EXPECT_EQ(response.Snapshots()[0], "snap1");
    EXPECT_EQ(response.Snapshots()[1], "snap2");
}

TEST_F(UnconnectMilvusMockedTest, ListSnapshotsFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListSnapshots(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListSnapshotsRequest*, ListSnapshotsResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::ListSnapshotsRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll");
    milvus::ListSnapshotsResponse response;
    auto status = client->ListSnapshots(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, ListSnapshotsEmptyCollectionNameIsAllowed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListSnapshots(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListSnapshotsRequest* request, ListSnapshotsResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_TRUE(request->collection_name().empty());
            response->add_snapshots("snap1");
            return ::grpc::Status{};
        });

    milvus::ListSnapshotsRequest request;
    request.WithDatabaseName("db");
    milvus::ListSnapshotsResponse response;
    auto status = client->ListSnapshots(request, response);

    EXPECT_TRUE(status.IsOk());
    ASSERT_EQ(response.Snapshots().size(), 1);
    EXPECT_EQ(response.Snapshots()[0], "snap1");
}

TEST_F(UnconnectMilvusMockedTest, DescribeSnapshot) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeSnapshot(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const DescribeSnapshotRequest* request, DescribeSnapshotResponse* response) {
                EXPECT_EQ(request->db_name(), "db");
                EXPECT_EQ(request->collection_name(), "coll");
                EXPECT_EQ(request->name(), "snap");
                response->set_name("snap");
                response->set_description("desc");
                response->set_collection_name("coll");
                response->add_partition_names("p1");
                response->add_partition_names("p2");
                response->set_create_ts(123);
                response->set_s3_location("s3://bucket/meta");
                return ::grpc::Status{};
            });

    milvus::DescribeSnapshotRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll").WithSnapshotName("snap");
    milvus::DescribeSnapshotResponse response;
    auto status = client->DescribeSnapshot(request, response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.Name(), "snap");
    EXPECT_EQ(response.Description(), "desc");
    EXPECT_EQ(response.CollectionName(), "coll");
    ASSERT_EQ(response.PartitionNames().size(), 2);
    EXPECT_EQ(response.PartitionNames()[0], "p1");
    EXPECT_EQ(response.CreateTs(), 123);
    EXPECT_EQ(response.S3Location(), "s3://bucket/meta");
}

TEST_F(UnconnectMilvusMockedTest, DescribeSnapshotInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::DescribeSnapshotRequest request;
    request.WithCollectionName("coll");
    milvus::DescribeSnapshotResponse response;
    auto status = client->DescribeSnapshot(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, DescribeSnapshotFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeSnapshot(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const DescribeSnapshotRequest*, DescribeSnapshotResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::DescribeSnapshotRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll").WithSnapshotName("snap");
    milvus::DescribeSnapshotResponse response;
    auto status = client->DescribeSnapshot(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, RestoreSnapshot) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, RestoreSnapshot(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const RestoreSnapshotRequest* request, RestoreSnapshotResponse* response) {
            EXPECT_EQ(request->name(), "snap");
            EXPECT_EQ(request->db_name(), "source_db");
            EXPECT_EQ(request->collection_name(), "source_coll");
            EXPECT_EQ(request->target_db_name(), "target_db");
            EXPECT_EQ(request->target_collection_name(), "target_coll");
            response->set_job_id(99);
            return ::grpc::Status{};
        });

    milvus::RestoreSnapshotRequest request;
    request.WithSnapshotName("snap")
        .WithSourceDatabaseName("source_db")
        .WithSourceCollectionName("source_coll")
        .WithTargetDatabaseName("target_db")
        .WithTargetCollectionName("target_coll");
    milvus::RestoreSnapshotResponse response;
    auto status = client->RestoreSnapshot(request, response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.JobID(), 99);
}

TEST_F(UnconnectMilvusMockedTest, RestoreSnapshotInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::RestoreSnapshotRequest request;
    request.WithSnapshotName("snap").WithSourceCollectionName("source_coll");
    milvus::RestoreSnapshotResponse response;
    auto status = client->RestoreSnapshot(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, RestoreSnapshotFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, RestoreSnapshot(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const RestoreSnapshotRequest*, RestoreSnapshotResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::RestoreSnapshotRequest request;
    request.WithSnapshotName("snap")
        .WithSourceDatabaseName("source_db")
        .WithSourceCollectionName("source_coll")
        .WithTargetDatabaseName("target_db")
        .WithTargetCollectionName("target_coll");
    milvus::RestoreSnapshotResponse response;
    auto status = client->RestoreSnapshot(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, GetRestoreSnapshotState) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetRestoreSnapshotState(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetRestoreSnapshotStateRequest* request,
                     GetRestoreSnapshotStateResponse* response) {
            EXPECT_EQ(request->job_id(), 101);
            FillRestoreSnapshotInfo(response->mutable_info());
            return ::grpc::Status{};
        });

    milvus::GetRestoreSnapshotStateRequest request;
    request.WithJobID(101);
    milvus::GetRestoreSnapshotStateResponse response;
    auto status = client->GetRestoreSnapshotState(request, response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.JobInfo().JobID(), 101);
    EXPECT_EQ(response.JobInfo().SnapshotName(), "snap");
    EXPECT_EQ(response.JobInfo().DatabaseName(), "db");
    EXPECT_EQ(response.JobInfo().CollectionName(), "coll");
    EXPECT_EQ(response.JobInfo().State(), milvus::RestoreSnapshotStateCode::COMPLETED);
    EXPECT_EQ(response.JobInfo().Progress(), 100);
    EXPECT_EQ(response.JobInfo().Reason(), "done");
    EXPECT_EQ(response.JobInfo().StartTime(), 1000);
    EXPECT_EQ(response.JobInfo().TimeCost(), 2000);
}

TEST_F(UnconnectMilvusMockedTest, GetRestoreSnapshotStateInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::GetRestoreSnapshotStateRequest request;
    milvus::GetRestoreSnapshotStateResponse response;
    auto status = client->GetRestoreSnapshotState(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, GetRestoreSnapshotStateFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetRestoreSnapshotState(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetRestoreSnapshotStateRequest*, GetRestoreSnapshotStateResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::GetRestoreSnapshotStateRequest request;
    request.WithJobID(101);
    milvus::GetRestoreSnapshotStateResponse response;
    auto status = client->GetRestoreSnapshotState(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, ListRestoreSnapshotJobs) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListRestoreSnapshotJobs(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListRestoreSnapshotJobsRequest* request,
                     ListRestoreSnapshotJobsResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collection_name(), "coll");
            FillRestoreSnapshotInfo(response->add_jobs());
            return ::grpc::Status{};
        });

    milvus::ListRestoreSnapshotJobsRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll");
    milvus::ListRestoreSnapshotJobsResponse response;
    auto status = client->ListRestoreSnapshotJobs(request, response);

    EXPECT_TRUE(status.IsOk());
    ASSERT_EQ(response.Jobs().size(), 1);
    EXPECT_EQ(response.Jobs()[0].JobID(), 101);
    EXPECT_EQ(response.Jobs()[0].SnapshotName(), "snap");
}

TEST_F(UnconnectMilvusMockedTest, ListRestoreSnapshotJobsFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListRestoreSnapshotJobs(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListRestoreSnapshotJobsRequest*, ListRestoreSnapshotJobsResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::ListRestoreSnapshotJobsRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll");
    milvus::ListRestoreSnapshotJobsResponse response;
    auto status = client->ListRestoreSnapshotJobs(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, ListRestoreSnapshotJobsEmptyCollectionNameIsAllowed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListRestoreSnapshotJobs(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListRestoreSnapshotJobsRequest* request,
                     ListRestoreSnapshotJobsResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_TRUE(request->collection_name().empty());
            FillRestoreSnapshotInfo(response->add_jobs());
            return ::grpc::Status{};
        });

    milvus::ListRestoreSnapshotJobsRequest request;
    request.WithDatabaseName("db");
    milvus::ListRestoreSnapshotJobsResponse response;
    auto status = client->ListRestoreSnapshotJobs(request, response);

    EXPECT_TRUE(status.IsOk());
    ASSERT_EQ(response.Jobs().size(), 1);
    EXPECT_EQ(response.Jobs()[0].JobID(), 101);
}

TEST_F(UnconnectMilvusMockedTest, PinSnapshotData) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, PinSnapshotData(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const PinSnapshotDataRequest* request, PinSnapshotDataResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_EQ(request->name(), "snap");
            EXPECT_EQ(request->ttl_seconds(), 60);
            response->set_pin_id(77);
            return ::grpc::Status{};
        });

    milvus::PinSnapshotDataRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll").WithSnapshotName("snap").WithTtlSeconds(60);
    milvus::PinSnapshotDataResponse response;
    auto status = client->PinSnapshotData(request, response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.PinID(), 77);
}

TEST_F(UnconnectMilvusMockedTest, PinSnapshotDataInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::PinSnapshotDataRequest request;
    request.WithCollectionName("coll");
    milvus::PinSnapshotDataResponse response;
    auto status = client->PinSnapshotData(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, PinSnapshotDataNegativeTtlInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::PinSnapshotDataRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll").WithSnapshotName("snap").WithTtlSeconds(-1);
    milvus::PinSnapshotDataResponse response;
    auto status = client->PinSnapshotData(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, PinSnapshotDataFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, PinSnapshotData(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const PinSnapshotDataRequest*, PinSnapshotDataResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::PinSnapshotDataRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll").WithSnapshotName("snap").WithTtlSeconds(60);
    milvus::PinSnapshotDataResponse response;
    auto status = client->PinSnapshotData(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, UnpinSnapshotData) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, UnpinSnapshotData(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const UnpinSnapshotDataRequest* request, ::milvus::proto::common::Status*) {
                EXPECT_EQ(request->pin_id(), 88);
                return ::grpc::Status{};
            });

    milvus::UnpinSnapshotDataRequest request;
    request.WithPinID(88);
    auto status = client->UnpinSnapshotData(request);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, UnpinSnapshotDataInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::UnpinSnapshotDataRequest request;
    auto status = client->UnpinSnapshotData(request);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, UnpinSnapshotDataFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, UnpinSnapshotData(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const UnpinSnapshotDataRequest*, ::milvus::proto::common::Status*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::UnpinSnapshotDataRequest request;
    request.WithPinID(88);
    auto status = client->UnpinSnapshotData(request);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}
