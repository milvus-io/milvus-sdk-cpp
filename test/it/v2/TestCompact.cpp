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

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"

using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::DescribeCollectionRequest;
using ::milvus::proto::milvus::DescribeCollectionResponse;
using ::milvus::proto::milvus::GetCompactionPlansRequest;
using ::milvus::proto::milvus::GetCompactionPlansResponse;
using ::milvus::proto::milvus::ManualCompactionRequest;
using ::milvus::proto::milvus::ManualCompactionResponse;
using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Property;

namespace {

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
