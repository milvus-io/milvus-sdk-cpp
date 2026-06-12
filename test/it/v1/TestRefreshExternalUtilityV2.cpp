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
using ::milvus::proto::milvus::AddFileResourceRequest;
using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::FileResourceInfo;
using ::milvus::proto::milvus::GetRefreshExternalCollectionProgressRequest;
using ::milvus::proto::milvus::GetRefreshExternalCollectionProgressResponse;
using ::milvus::proto::milvus::ListFileResourcesRequest;
using ::milvus::proto::milvus::ListFileResourcesResponse;
using ::milvus::proto::milvus::ListRefreshExternalCollectionJobsRequest;
using ::milvus::proto::milvus::ListRefreshExternalCollectionJobsResponse;
using ::milvus::proto::milvus::RefreshExternalCollectionJobInfo;
using ::milvus::proto::milvus::RefreshExternalCollectionRequest;
using ::milvus::proto::milvus::RefreshExternalCollectionResponse;
using ::milvus::proto::milvus::RefreshExternalCollectionState;
using ::milvus::proto::milvus::RemoveFileResourceRequest;
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
