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

#include <memory>
#include <stdexcept>

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"
#include "milvus/types/ConnectParam.h"
#include "utils/ConnectionHandler.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::AddFileResourceRequest;
using ::milvus::proto::milvus::CancelRequestsRequest;
using ::milvus::proto::milvus::CancelRequestsResponse;
using ::milvus::proto::milvus::CheckHealthRequest;
using ::milvus::proto::milvus::CheckHealthResponse;
using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::DescribeCollectionRequest;
using ::milvus::proto::milvus::DescribeCollectionResponse;
using ::milvus::proto::milvus::FileResourceInfo;
using ::milvus::proto::milvus::FlushAllRequest;
using ::milvus::proto::milvus::FlushAllResponse;
using ::milvus::proto::milvus::GetCompactionPlansRequest;
using ::milvus::proto::milvus::GetCompactionPlansResponse;
using ::milvus::proto::milvus::GetFlushAllStateRequest;
using ::milvus::proto::milvus::GetFlushAllStateResponse;
using ::milvus::proto::milvus::GetRefreshExternalCollectionProgressRequest;
using ::milvus::proto::milvus::GetRefreshExternalCollectionProgressResponse;
using ::milvus::proto::milvus::GetVersionRequest;
using ::milvus::proto::milvus::GetVersionResponse;
using ::milvus::proto::milvus::ListFileResourcesRequest;
using ::milvus::proto::milvus::ListFileResourcesResponse;
using ::milvus::proto::milvus::ListRefreshExternalCollectionJobsRequest;
using ::milvus::proto::milvus::ListRefreshExternalCollectionJobsResponse;
using ::milvus::proto::milvus::ListRunningRequestsRequest;
using ::milvus::proto::milvus::ListRunningRequestsResponse;
using ::milvus::proto::milvus::ManualCompactionRequest;
using ::milvus::proto::milvus::ManualCompactionResponse;
using ::milvus::proto::milvus::RefreshExternalCollectionJobInfo;
using ::milvus::proto::milvus::RefreshExternalCollectionRequest;
using ::milvus::proto::milvus::RefreshExternalCollectionResponse;
using ::milvus::proto::milvus::RefreshExternalCollectionState;
using ::milvus::proto::milvus::RemoveFileResourceRequest;
using ::milvus::proto::milvus::RunningRequestInfo;
using ::milvus::proto::milvus::RunningRequestNodeResult;
using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Property;

namespace {

milvus::Status
ConnectHandler(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port,
               milvus::ConnectionHandler& handler) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) { return ::grpc::Status{}; });
    return handler.Connect(milvus::ConnectParam{"127.0.0.1", port});
}

std::shared_ptr<milvus::MilvusClientV2>
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
ExpectDescribeCollection(testing::StrictMock<::milvus::MilvusMockedService>& service) {
    EXPECT_CALL(service, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
            response->set_collectionid(200);
            return ::grpc::Status{};
        });
}

void
FillRefreshJobInfo(RefreshExternalCollectionJobInfo* info) {
    info->set_job_id(101);
    info->set_collection_name("coll");
    info->set_state(RefreshExternalCollectionState::RefreshCompleted);
    info->set_progress(100);
    info->set_reason("done");
    info->set_external_source("s3://bucket/path/");
    info->set_start_time(1000);
    info->set_end_time(2000);
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, CompactWithIsL0AndTargetSizeUnit) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    ExpectDescribeCollection(service_);

    EXPECT_CALL(service_, ManualCompaction(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const ManualCompactionRequest* request, ManualCompactionResponse* response) {
                EXPECT_EQ(request->collectionid(), 200);
                EXPECT_TRUE(request->majorcompaction());
                EXPECT_TRUE(request->l0compaction());
                // 1GB = 1024MB
                EXPECT_EQ(request->target_size(), 1024);
                response->set_compactionid(10);
                return ::grpc::Status{};
            });

    milvus::CompactResponse response;
    auto status = client->Compact(milvus::CompactRequest()
                                      .WithDatabaseName("db")
                                      .WithCollectionName("collection")
                                      .WithClusteringCompaction(true)
                                      .WithIsL0(true)
                                      .WithTargetSize(1)
                                      .WithTargetSizeUnit("gb"),
                                  response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.CompactionID(), 10);
}

TEST_F(UnconnectMilvusMockedTest, CompactDefaultUnitMb) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    ExpectDescribeCollection(service_);

    EXPECT_CALL(service_, ManualCompaction(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const ManualCompactionRequest* request, ManualCompactionResponse* response) {
                EXPECT_EQ(request->target_size(), 512);
                EXPECT_FALSE(request->l0compaction());
                response->set_compactionid(11);
                return ::grpc::Status{};
            });

    milvus::CompactResponse response;
    auto status = client->Compact(
        milvus::CompactRequest().WithDatabaseName("db").WithCollectionName("collection").WithTargetSize(512), response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.CompactionID(), 11);
}

TEST_F(UnconnectMilvusMockedTest, CompactRejectsNegativeTargetSize) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    ExpectDescribeCollection(service_);

    milvus::CompactResponse response;
    auto status = client->Compact(
        milvus::CompactRequest().WithDatabaseName("db").WithCollectionName("collection").WithTargetSize(-1), response);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, CompactRejectsInvalidUnit) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    ExpectDescribeCollection(service_);

    milvus::CompactResponse response;
    auto status = client->Compact(milvus::CompactRequest()
                                      .WithDatabaseName("db")
                                      .WithCollectionName("collection")
                                      .WithTargetSize(1)
                                      .WithTargetSizeUnit("xq"),
                                  response);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, GetCompactionPlansV2) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    const int64_t compaction_id = 42;
    const std::vector<int64_t> sources = {1, 2, 3};
    const int64_t target = 100;

    EXPECT_CALL(service_,
                GetCompactionStateWithPlans(_, Property(&GetCompactionPlansRequest::compactionid, compaction_id), _))
        .WillOnce([&](::grpc::ServerContext*, const GetCompactionPlansRequest*, GetCompactionPlansResponse* response) {
            response->set_state(milvus::proto::common::CompactionState::Completed);
            auto info = response->add_mergeinfos();
            for (auto i : sources) {
                info->add_sources(i);
            }
            info->set_target(target);
            info->set_plan_id(11);
            info->set_trigger_id(12);
            info->set_collection_id(13);
            info->set_partition_id(14);
            info->set_channel("ch-0");
            info->set_type(milvus::proto::common::CompactionType::CompactionTypeMajor);
            info->set_state(milvus::proto::common::CompactionTaskState::CompactionTaskStateCompleted);
            info->set_failure_reason("boom");
            info->add_targets(target + 1);
            return ::grpc::Status{};
        });

    milvus::GetCompactionPlansResponse response;
    auto status =
        client->GetCompactionPlans(milvus::GetCompactionPlansRequest().WithCompactionID(compaction_id), response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.CompactionID(), compaction_id);
    EXPECT_EQ(response.State(), milvus::CompactionStateCode::COMPLETED);
    ASSERT_EQ(response.Plans().size(), 1);
    EXPECT_THAT(response.Plans()[0].SourceSegments(), ElementsAreArray(sources));
    EXPECT_EQ(response.Plans()[0].DestinySegemnt(), target);
    EXPECT_EQ(response.Plans()[0].PlanId(), 11);
    EXPECT_EQ(response.Plans()[0].TriggerId(), 12);
    EXPECT_EQ(response.Plans()[0].CollectionId(), 13);
    EXPECT_EQ(response.Plans()[0].PartitionId(), 14);
    EXPECT_EQ(response.Plans()[0].Channel(), "ch-0");
    EXPECT_EQ(response.Plans()[0].Type(), milvus::CompactionType::MAJOR);
    EXPECT_EQ(response.Plans()[0].State(), milvus::CompactionTaskState::COMPLETED);
    EXPECT_EQ(response.Plans()[0].FailureReason(), "boom");
    EXPECT_THAT(response.Plans()[0].Targets(), ElementsAreArray(std::vector<int64_t>{target + 1}));
}

TEST_F(UnconnectMilvusMockedTest, GetCompactionPlansV2ExecutingState) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    const int64_t compaction_id = 43;

    EXPECT_CALL(service_,
                GetCompactionStateWithPlans(_, Property(&GetCompactionPlansRequest::compactionid, compaction_id), _))
        .WillOnce([](::grpc::ServerContext*, const GetCompactionPlansRequest*, GetCompactionPlansResponse* response) {
            response->set_state(milvus::proto::common::CompactionState::Executing);
            return ::grpc::Status{};
        });

    milvus::GetCompactionPlansResponse response;
    auto status =
        client->GetCompactionPlans(milvus::GetCompactionPlansRequest().WithCompactionID(compaction_id), response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.CompactionID(), compaction_id);
    EXPECT_EQ(response.State(), milvus::CompactionStateCode::EXECUTING);
}

TEST_F(UnconnectMilvusMockedTest, GetCompactionPlansV2UnsetState) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    const int64_t compaction_id = 44;

    EXPECT_CALL(service_,
                GetCompactionStateWithPlans(_, Property(&GetCompactionPlansRequest::compactionid, compaction_id), _))
        .WillOnce([](::grpc::ServerContext*, const GetCompactionPlansRequest*, GetCompactionPlansResponse*) {
            return ::grpc::Status{};
        });

    milvus::GetCompactionPlansResponse response;
    auto status =
        client->GetCompactionPlans(milvus::GetCompactionPlansRequest().WithCompactionID(compaction_id), response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.CompactionID(), compaction_id);
    EXPECT_EQ(response.State(), milvus::CompactionStateCode::UNKNOWN);
}

TEST_F(UnconnectMilvusMockedTest, ListCompactionTasksV2) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    const std::string db_name = "test_db";
    const std::string collection_name = "test_collection";
    const std::vector<int64_t> sources = {1, 2, 3};
    const int64_t target = 100;

    EXPECT_CALL(service_, GetCompactionStateWithPlans(
                              _, Property(&GetCompactionPlansRequest::collection_name, collection_name), _))
        .WillOnce([&](::grpc::ServerContext*, const GetCompactionPlansRequest*, GetCompactionPlansResponse* response) {
            response->set_state(milvus::proto::common::CompactionState::Completed);
            auto info = response->add_mergeinfos();
            for (auto i : sources) {
                info->add_sources(i);
            }
            info->set_target(target);
            info->set_plan_id(11);
            info->set_trigger_id(12);
            info->set_collection_id(13);
            info->set_partition_id(14);
            info->set_channel("ch-0");
            info->set_type(milvus::proto::common::CompactionType::CompactionTypeLevel0Delete);
            info->set_state(milvus::proto::common::CompactionTaskState::CompactionTaskStateFailed);
            info->set_failure_reason("boom");
            info->add_targets(target + 1);
            return ::grpc::Status{};
        });

    milvus::GetCompactionPlansResponse response;
    auto status = client->ListCompactionTasks(
        milvus::ListCompactionTasksRequest().WithDatabaseName(db_name).WithCollectionName(collection_name), response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.CollectionName(), collection_name);
    EXPECT_EQ(response.State(), milvus::CompactionStateCode::COMPLETED);
    ASSERT_EQ(response.Plans().size(), 1);
    EXPECT_THAT(response.Plans()[0].SourceSegments(), ElementsAreArray(sources));
    EXPECT_EQ(response.Plans()[0].DestinySegemnt(), target);
    EXPECT_EQ(response.Plans()[0].PlanId(), 11);
    EXPECT_EQ(response.Plans()[0].TriggerId(), 12);
    EXPECT_EQ(response.Plans()[0].CollectionId(), 13);
    EXPECT_EQ(response.Plans()[0].PartitionId(), 14);
    EXPECT_EQ(response.Plans()[0].Channel(), "ch-0");
    EXPECT_EQ(response.Plans()[0].Type(), milvus::CompactionType::LEVEL0_DELETE);
    EXPECT_EQ(response.Plans()[0].State(), milvus::CompactionTaskState::FAILED);
    EXPECT_EQ(response.Plans()[0].FailureReason(), "boom");
    EXPECT_THAT(response.Plans()[0].Targets(), ElementsAreArray(std::vector<int64_t>{target + 1}));
}
TEST_F(UnconnectMilvusMockedTest, FlushAll) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, FlushAll(_, Property(&FlushAllRequest::db_name, "db1"), _))
        .WillOnce([](::grpc::ServerContext*, const FlushAllRequest*, FlushAllResponse* response) {
            response->set_flush_all_ts(12345);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, GetFlushAllState(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const GetFlushAllStateRequest* request, GetFlushAllStateResponse* response) {
                EXPECT_EQ(request->db_name(), "db1");
                EXPECT_EQ(request->flush_all_ts(), 12345);
                response->set_flushed(true);
                return ::grpc::Status{};
            });

    milvus::FlushAllResponse response;
    auto status = client->FlushAll(milvus::FlushAllRequest().WithDatabaseName("db1"), response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.FlushAllTs(), 12345);
}

TEST_F(UnconnectMilvusMockedTest, FlushAllFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, FlushAll(_, Property(&FlushAllRequest::db_name, "db1"), _))
        .WillOnce([](::grpc::ServerContext*, const FlushAllRequest*, FlushAllResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::FlushAllResponse response;
    auto status = client->FlushAll(milvus::FlushAllRequest().WithDatabaseName("db1"), response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, FlushAllServerFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, FlushAll(_, Property(&FlushAllRequest::db_name, "db1"), _))
        .WillOnce([](::grpc::ServerContext*, const FlushAllRequest*, FlushAllResponse* response) {
            response->mutable_status()->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
            return ::grpc::Status{};
        });

    milvus::FlushAllResponse response;
    auto status = client->FlushAll(milvus::FlushAllRequest().WithDatabaseName("db1"), response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, GetFlushAllState) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetFlushAllState(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const GetFlushAllStateRequest* request, GetFlushAllStateResponse* response) {
                EXPECT_EQ(request->db_name(), "db1");
                EXPECT_EQ(request->flush_all_ts(), 12345);
                response->set_flushed(true);
                return ::grpc::Status{};
            });

    milvus::GetFlushAllStateResponse response;
    auto status = client->GetFlushAllState(
        milvus::GetFlushAllStateRequest().WithDatabaseName("db1").WithFlushAllTs(12345), response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_TRUE(response.Flushed());
}

TEST_F(UnconnectMilvusMockedTest, GetFlushAllStateFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetFlushAllState(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetFlushAllStateRequest*, GetFlushAllStateResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::GetFlushAllStateResponse response;
    auto status = client->GetFlushAllState(
        milvus::GetFlushAllStateRequest().WithDatabaseName("db1").WithFlushAllTs(12345), response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, GetFlushAllStateServerFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetFlushAllState(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetFlushAllStateRequest*, GetFlushAllStateResponse* response) {
            response->mutable_status()->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
            return ::grpc::Status{};
        });

    milvus::GetFlushAllStateResponse response;
    auto status = client->GetFlushAllState(
        milvus::GetFlushAllStateRequest().WithDatabaseName("db1").WithFlushAllTs(12345), response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}
TEST_F(UnconnectMilvusMockedTest, GetServerVersionV2WithoutDetail) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetVersion(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetVersionRequest*, GetVersionResponse* response) {
            response->set_version("v2.5.0");
            return ::grpc::Status{};
        });

    milvus::GetServerVersionResponse response;
    auto status = client->GetServerVersionV2(milvus::GetServerVersionRequest(), response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.Version(), "v2.5.0");
    EXPECT_TRUE(response.BuildTime().empty());
}

TEST_F(UnconnectMilvusMockedTest, GetServerVersionV2WithDetail) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    // The second Connect RPC is issued for the detailed version query; it should carry
    // the SDK client_info for parity with the normal connect handshake.
    EXPECT_CALL(service_, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest* request, ConnectResponse* response) {
            EXPECT_EQ(request->client_info().sdk_type(), "CPP");
            EXPECT_FALSE(request->client_info().sdk_version().empty());
            auto* info = response->mutable_server_info();
            info->set_build_tags("v2.5.0");
            info->set_build_time("2024-01-01 00:00:00");
            info->set_git_commit("abc123");
            info->set_go_version("go1.21");
            info->set_deploy_mode("standalone");
            return ::grpc::Status{};
        });

    milvus::GetServerVersionResponse response;
    auto status = client->GetServerVersionV2(milvus::GetServerVersionRequest().WithDetail(true), response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.Version(), "v2.5.0");
    EXPECT_EQ(response.BuildTime(), "2024-01-01 00:00:00");
    EXPECT_EQ(response.GitCommit(), "abc123");
    EXPECT_EQ(response.GoVersion(), "go1.21");
    EXPECT_EQ(response.DeployMode(), "standalone");
}
TEST_F(UnconnectMilvusMockedTest, ExceptionBarrierConvertsStdExceptionToStatus) {
    milvus::ConnectionHandler handler;
    auto connect_status = ConnectHandler(service_, server_.ListenPort(), handler);
    ASSERT_TRUE(connect_status.IsOk());

    auto throwing_validate = []() -> milvus::Status { throw std::runtime_error("boom"); };
    auto pre = [](CheckHealthRequest&) { return milvus::Status::OK(); };
    auto post = [](const CheckHealthResponse&) { return milvus::Status::OK(); };

    auto status = handler.Invoke<CheckHealthRequest, CheckHealthResponse>(throwing_validate, pre,
                                                                          &milvus::MilvusConnection::CheckHealth, post);

    EXPECT_EQ(status.Code(), milvus::StatusCode::UNKNOWN_ERROR);
    EXPECT_NE(status.Message().find("Unexpected SDK exception: boom"), std::string::npos);
}

TEST_F(UnconnectMilvusMockedTest, ExceptionBarrierConvertsNonStdExceptionToStatus) {
    milvus::ConnectionHandler handler;
    auto connect_status = ConnectHandler(service_, server_.ListenPort(), handler);
    ASSERT_TRUE(connect_status.IsOk());

    auto throwing_validate = []() -> milvus::Status { throw 42; };
    auto pre = [](CheckHealthRequest&) { return milvus::Status::OK(); };
    auto post = [](const CheckHealthResponse&) { return milvus::Status::OK(); };

    auto status = handler.Invoke<CheckHealthRequest, CheckHealthResponse>(throwing_validate, pre,
                                                                          &milvus::MilvusConnection::CheckHealth, post);

    EXPECT_EQ(status.Code(), milvus::StatusCode::UNKNOWN_ERROR);
    EXPECT_NE(status.Message().find("Unexpected SDK exception"), std::string::npos);
}
TEST_F(UnconnectMilvusMockedTest, RefreshExternalCollection) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, RefreshExternalCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const RefreshExternalCollectionRequest* request,
                     RefreshExternalCollectionResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_EQ(request->external_source(), "s3://bucket/path/");
            EXPECT_EQ(request->external_spec(), "{\"format\":\"parquet\"}");
            response->set_job_id(77);
            return ::grpc::Status{};
        });

    milvus::RefreshExternalCollectionRequest request;
    request.WithDatabaseName("db")
        .WithCollectionName("coll")
        .WithExternalSource("s3://bucket/path/")
        .WithExternalSpec(nlohmann::json{{"format", "parquet"}});
    milvus::RefreshExternalCollectionResponse response;
    auto status = client->RefreshExternalCollection(request, response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.JobID(), 77);
}

TEST_F(UnconnectMilvusMockedTest, RefreshExternalCollectionInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::RefreshExternalCollectionRequest request;
    milvus::RefreshExternalCollectionResponse response;
    auto status = client->RefreshExternalCollection(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, GetRefreshExternalCollectionProgress) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetRefreshExternalCollectionProgress(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetRefreshExternalCollectionProgressRequest* request,
                     GetRefreshExternalCollectionProgressResponse* response) {
            EXPECT_EQ(request->job_id(), 101);
            FillRefreshJobInfo(response->mutable_job_info());
            return ::grpc::Status{};
        });

    milvus::GetRefreshExternalCollectionProgressRequest request;
    request.WithJobID(101);
    milvus::GetRefreshExternalCollectionProgressResponse response;
    auto status = client->GetRefreshExternalCollectionProgress(request, response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.JobInfo().JobID(), 101);
    EXPECT_EQ(response.JobInfo().CollectionName(), "coll");
    EXPECT_EQ(response.JobInfo().Progress(), 100);
    EXPECT_EQ(response.JobInfo().ExternalSource(), "s3://bucket/path/");
}

TEST_F(UnconnectMilvusMockedTest, GetRefreshExternalCollectionProgressInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::GetRefreshExternalCollectionProgressRequest request;
    milvus::GetRefreshExternalCollectionProgressResponse response;
    auto status = client->GetRefreshExternalCollectionProgress(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, ListRefreshExternalCollectionJobs) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListRefreshExternalCollectionJobs(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListRefreshExternalCollectionJobsRequest* request,
                     ListRefreshExternalCollectionJobsResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collection_name(), "coll");
            FillRefreshJobInfo(response->add_jobs());
            return ::grpc::Status{};
        });

    milvus::ListRefreshExternalCollectionJobsRequest request;
    request.WithDatabaseName("db").WithCollectionName("coll");
    milvus::ListRefreshExternalCollectionJobsResponse response;
    auto status = client->ListRefreshExternalCollectionJobs(request, response);

    EXPECT_TRUE(status.IsOk());
    ASSERT_EQ(response.Jobs().size(), 1);
    EXPECT_EQ(response.Jobs()[0].JobID(), 101);
}

TEST_F(UnconnectMilvusMockedTest, AddFileResource) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, AddFileResource(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AddFileResourceRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->name(), "res1");
            EXPECT_EQ(request->path(), "/tmp/data.parquet");
            return ::grpc::Status{};
        });

    milvus::AddFileResourceRequest request;
    request.WithName("res1").WithPath("/tmp/data.parquet");
    auto status = client->AddFileResource(request);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, AddFileResourceInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::AddFileResourceRequest request;
    request.WithName("res1");
    auto status = client->AddFileResource(request);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, RemoveFileResource) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, RemoveFileResource(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const RemoveFileResourceRequest* request, ::milvus::proto::common::Status*) {
                EXPECT_EQ(request->name(), "res1");
                return ::grpc::Status{};
            });

    milvus::RemoveFileResourceRequest request;
    request.WithName("res1");
    auto status = client->RemoveFileResource(request);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, RemoveFileResourceInvalidArgument) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::RemoveFileResourceRequest request;
    auto status = client->RemoveFileResource(request);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, ListFileResources) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListFileResources(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListFileResourcesRequest*, ListFileResourcesResponse* response) {
            auto* resource = response->add_resources();
            resource->set_name("res1");
            resource->set_path("/tmp/data.parquet");
            return ::grpc::Status{};
        });

    milvus::ListFileResourcesRequest request;
    milvus::ListFileResourcesResponse response;
    auto status = client->ListFileResources(request, response);

    EXPECT_TRUE(status.IsOk());
    ASSERT_EQ(response.Resources().size(), 1);
    EXPECT_EQ(response.Resources()[0].Name(), "res1");
    EXPECT_EQ(response.Resources()[0].Path(), "/tmp/data.parquet");
}

namespace {
void
FillGrpcRunningRequest(RunningRequestInfo* info, const std::string& id, const std::string& state) {
    info->set_request_id(id);
    info->set_proxy_id(7);
    info->set_type("Search");
    info->set_db_name("db");
    info->set_collection_name("coll");
    info->set_user("alice");
    info->set_client_addr("10.0.0.1:5000");
    info->set_nq(4000);
    info->set_topk(100);
    info->set_expr("id > 0");
    info->set_start_time_ms(1700000000000);
    info->set_queued_ms(3);
    info->set_elapsed_ms(1500);
    info->set_state(state);
    info->add_task_ids(11);
    info->add_task_ids(12);
    info->set_trace_id("trace-1");
    info->set_cancellable(true);
}
}  // namespace

TEST_F(UnconnectMilvusMockedTest, ListRunningRequestsV2) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListRunningRequests(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const ListRunningRequestsRequest* request,
                      ListRunningRequestsResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_EQ(request->user(), "alice");
            EXPECT_EQ(request->min_elapsed_ms(), 1000);

            FillGrpcRunningRequest(response->add_requests(), "7-to03zu-1", "Running");
            auto* answered = response->add_node_results();
            answered->set_node_id(7);
            auto* unimplemented = response->add_node_results();
            unimplemented->set_node_id(9);
            unimplemented->mutable_status()->set_reason("unimplemented");
            unimplemented->mutable_status()->set_error_code(::milvus::proto::common::ErrorCode::UnexpectedError);
            unimplemented->set_unimplemented(true);
            return ::grpc::Status{};
        });

    milvus::ListRunningRequestsResponse response;
    auto status = client->ListRunningRequests(milvus::ListRunningRequestsRequest()
                                                  .WithDatabaseName("db")
                                                  .WithCollectionName("coll")
                                                  .WithUser("alice")
                                                  .WithMinElapsedMs(1000),
                                              response);

    EXPECT_TRUE(status.IsOk());
    ASSERT_EQ(response.Requests().size(), 1);
    const auto& info = response.Requests()[0];
    EXPECT_EQ(info.RequestId(), "7-to03zu-1");
    EXPECT_EQ(info.ProxyId(), 7);
    EXPECT_EQ(info.Type(), "Search");
    EXPECT_EQ(info.DatabaseName(), "db");
    EXPECT_EQ(info.CollectionName(), "coll");
    EXPECT_EQ(info.User(), "alice");
    EXPECT_EQ(info.ClientAddr(), "10.0.0.1:5000");
    EXPECT_EQ(info.Nq(), 4000);
    EXPECT_EQ(info.TopK(), 100);
    EXPECT_EQ(info.Expr(), "id > 0");
    EXPECT_EQ(info.StartTimeMs(), 1700000000000);
    EXPECT_EQ(info.QueuedMs(), 3);
    EXPECT_EQ(info.ElapsedMs(), 1500);
    EXPECT_EQ(info.State(), "Running");
    EXPECT_EQ(info.TaskIds().size(), 2);
    EXPECT_EQ(info.TaskIds()[0], 11);
    EXPECT_EQ(info.TaskIds()[1], 12);
    EXPECT_EQ(info.TraceId(), "trace-1");
    EXPECT_TRUE(info.Cancellable());

    ASSERT_EQ(response.NodeResults().size(), 2);
    EXPECT_EQ(response.NodeResults()[0].NodeId(), 7);
    EXPECT_TRUE(response.NodeResults()[0].IsSuccess());
    EXPECT_FALSE(response.NodeResults()[0].Unimplemented());
    EXPECT_EQ(response.NodeResults()[1].NodeId(), 9);
    EXPECT_FALSE(response.NodeResults()[1].IsSuccess());
    EXPECT_EQ(response.NodeResults()[1].ErrorCode(), ::milvus::proto::common::ErrorCode::UnexpectedError);
    EXPECT_EQ(response.NodeResults()[1].Reason(), "unimplemented");
    EXPECT_TRUE(response.NodeResults()[1].Unimplemented());
}

TEST_F(UnconnectMilvusMockedTest, ListRunningRequestsWithoutFiltersV2) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListRunningRequests(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const ListRunningRequestsRequest* request,
                      ListRunningRequestsResponse* response) {
            EXPECT_TRUE(request->db_name().empty());
            EXPECT_TRUE(request->collection_name().empty());
            EXPECT_TRUE(request->user().empty());
            EXPECT_EQ(request->min_elapsed_ms(), 0);
            return ::grpc::Status{};
        });

    milvus::ListRunningRequestsResponse response;
    auto status = client->ListRunningRequests(milvus::ListRunningRequestsRequest{}, response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_TRUE(response.Requests().empty());
    EXPECT_TRUE(response.NodeResults().empty());
}

TEST_F(UnconnectMilvusMockedTest, ListRunningRequestsServerFailedV2) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListRunningRequests(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListRunningRequestsRequest*, ListRunningRequestsResponse* response) {
            response->mutable_status()->set_code(1100);
            response->mutable_status()->set_reason("permission denied");
            return ::grpc::Status{};
        });

    milvus::ListRunningRequestsResponse response;
    auto status = client->ListRunningRequests(milvus::ListRunningRequestsRequest{}, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, CancelRequestsV2) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, CancelRequests(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const CancelRequestsRequest* request, CancelRequestsResponse* response) {
            EXPECT_EQ(request->request_ids_size(), 3);
            EXPECT_EQ(request->request_ids(0), "7-to03zu-1");
            EXPECT_EQ(request->request_ids(1), "7-to03zu-2");
            EXPECT_EQ(request->request_ids(2), "8-to03zv-3");
            EXPECT_EQ(request->reason(), "too heavy");
            EXPECT_EQ(request->user(), "alice");

            FillGrpcRunningRequest(response->add_canceled(), "7-to03zu-1", "Running");
            response->add_not_found("7-to03zu-2");
            response->add_undetermined("8-to03zv-3");
            auto* answered = response->add_node_results();
            answered->set_node_id(7);
            auto* silent = response->add_node_results();
            silent->set_node_id(8);
            silent->mutable_status()->set_code(65535);
            silent->mutable_status()->set_reason("node unreachable");
            return ::grpc::Status{};
        });

    milvus::CancelRequestsResponse response;
    auto status =
        client->CancelRequests(milvus::CancelRequestsRequest()
                                   .WithRequestIds(std::vector<std::string>{"7-to03zu-1", "7-to03zu-2", "8-to03zv-3"})
                                   .WithReason("too heavy")
                                   .WithUser("alice"),
                               response);

    EXPECT_TRUE(status.IsOk());
    ASSERT_EQ(response.Canceled().size(), 1);
    EXPECT_EQ(response.Canceled()[0].RequestId(), "7-to03zu-1");
    EXPECT_EQ(response.Canceled()[0].State(), "Running");
    EXPECT_EQ(response.NotFound().size(), 1);
    EXPECT_EQ(response.NotFound()[0], "7-to03zu-2");
    EXPECT_EQ(response.Undetermined().size(), 1);
    EXPECT_EQ(response.Undetermined()[0], "8-to03zv-3");

    ASSERT_EQ(response.NodeResults().size(), 2);
    EXPECT_TRUE(response.NodeResults()[0].IsSuccess());
    EXPECT_FALSE(response.NodeResults()[1].IsSuccess());
    EXPECT_EQ(response.NodeResults()[1].ErrorCode(), 65535);
    EXPECT_EQ(response.NodeResults()[1].Reason(), "node unreachable");
}

TEST_F(UnconnectMilvusMockedTest, CancelRequestsRejectsEmptyIds) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::CancelRequestsResponse response;
    auto status = client->CancelRequests(milvus::CancelRequestsRequest{}, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, CancelRequestsServerFailedV2) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, CancelRequests(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const CancelRequestsRequest*, CancelRequestsResponse* response) {
            response->mutable_status()->set_code(1100);
            response->mutable_status()->set_reason("permission denied");
            return ::grpc::Status{};
        });

    milvus::CancelRequestsResponse response;
    auto status = client->CancelRequests(
        milvus::CancelRequestsRequest().WithRequestIds(std::vector<std::string>{"7-to03zu-1"}), response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}
