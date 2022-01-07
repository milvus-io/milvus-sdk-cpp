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

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::DescribeCollectionRequest;
using ::milvus::proto::milvus::DescribeCollectionResponse;
using ::milvus::proto::milvus::GetCompactionPlansRequest;
using ::milvus::proto::milvus::GetCompactionPlansResponse;
using ::milvus::proto::milvus::GetCompactionStateRequest;
using ::milvus::proto::milvus::GetCompactionStateResponse;
using ::milvus::proto::milvus::ManualCompactionRequest;
using ::milvus::proto::milvus::ManualCompactionResponse;
using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, GetCompactionStateFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const int64_t compaction_id = 1;
    const int64_t executing_id = 100;
    const int64_t timeout_id = 101;
    const int64_t completed_id = 102;

    EXPECT_CALL(service_, GetCompactionState(_, Property(&GetCompactionStateRequest::compactionid, compaction_id), _))
        .WillOnce([](::grpc::ServerContext*, const GetCompactionStateRequest*, GetCompactionStateResponse* response) {
            response->set_executingplanno(executing_id);
            response->set_timeoutplanno(timeout_id);
            response->set_completedplanno(completed_id);
            response->set_state(::milvus::proto::common::CompactionState::Executing);
            return ::grpc::Status{};
        });

    ::milvus::CompactionState state;
    auto status = client_->GetCompactionState(compaction_id, state);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(state.State(), ::milvus::CompactionStateCode::EXECUTING);
    EXPECT_EQ(state.ExecutingPlan(), executing_id);
    EXPECT_EQ(state.TimeoutPlan(), timeout_id);
    EXPECT_EQ(state.CompletedPlan(), completed_id);

    EXPECT_CALL(service_, GetCompactionState(_, Property(&GetCompactionStateRequest::compactionid, compaction_id), _))
        .WillOnce([](::grpc::ServerContext*, const GetCompactionStateRequest*, GetCompactionStateResponse* response) {
            response->set_state(::milvus::proto::common::CompactionState::Completed);
            return ::grpc::Status{};
        });

    status = client_->GetCompactionState(compaction_id, state);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(state.State(), ::milvus::CompactionStateCode::COMPLETED);
}

TEST_F(MilvusMockedTest, GetCompactionStateFooConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    const int64_t compaction_id = 1;
    ::milvus::CompactionState state;
    auto status = client_->GetCompactionState(compaction_id, state);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, GetCompactionStateFooFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const int64_t compaction_id = 1;

    EXPECT_CALL(service_, GetCompactionState(_, Property(&GetCompactionStateRequest::compactionid, compaction_id), _))
        .WillOnce([](::grpc::ServerContext*, const GetCompactionStateRequest*, GetCompactionStateResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
        });

    ::milvus::CompactionState state;
    auto status = client_->GetCompactionState(compaction_id, state);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(MilvusMockedTest, ManualCompactionFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const int64_t compaction_id = 1;
    const std::string collection_name = "test";
    const int64_t collection_id = 9;
    const uint64_t travel_ts = 100;

    EXPECT_CALL(service_,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillOnce([](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
            response->set_collectionid(collection_id);
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, ManualCompaction(_,
                                           AllOf(Property(&ManualCompactionRequest::collectionid, collection_id),
                                                 Property(&ManualCompactionRequest::timetravel, travel_ts)),
                                           _))
        .WillOnce([](::grpc::ServerContext*, const ManualCompactionRequest*, ManualCompactionResponse* response) {
            response->set_compactionid(compaction_id);
            return ::grpc::Status{};
        });

    int64_t returned_compaction_id = 0;
    auto status = client_->ManualCompaction(collection_name, travel_ts, returned_compaction_id);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(returned_compaction_id, compaction_id);
}

TEST_F(MilvusMockedTest, ManualCompactionFooWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    const std::string collection_name = "test";
    const uint64_t travel_ts = 100;
    int64_t returned_compaction_id = 0;
    auto status = client_->ManualCompaction(collection_name, travel_ts, returned_compaction_id);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, ManualCompactionFooFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection_name = "test";
    const uint64_t travel_ts = 100;

    EXPECT_CALL(service_,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillOnce([](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
        });

    int64_t returned_compaction_id = 0;
    auto status = client_->ManualCompaction(collection_name, travel_ts, returned_compaction_id);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
TEST_F(MilvusMockedTest, GetCompactionPlansFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const int64_t compaction_id = 1;
    const std::vector<int64_t> sources = {1, 2, 3, 4};
    const int64_t target = 100;

    EXPECT_CALL(service_,
                GetCompactionStateWithPlans(_, Property(&GetCompactionPlansRequest::compactionid, compaction_id), _))
        .WillOnce([&sources, &target](::grpc::ServerContext*, const GetCompactionPlansRequest*,
                                      GetCompactionPlansResponse* response) {
            auto info = response->add_mergeinfos();
            for (auto i : sources) {
                info->add_sources(i);
            }
            info->set_target(target);

            return ::grpc::Status{};
        });

    ::milvus::CompactionPlans plans;
    auto status = client_->GetCompactionPlans(compaction_id, plans);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(1, plans.size());
    EXPECT_THAT(plans[0].SourceSegments(), ElementsAreArray(sources));
}

TEST_F(MilvusMockedTest, GetCompactionPlansFooConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    const int64_t compaction_id = 1;
    ::milvus::CompactionPlans plans;
    auto status = client_->GetCompactionPlans(compaction_id, plans);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, GetCompactionPlansFooFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const int64_t compaction_id = 1;

    EXPECT_CALL(service_,
                GetCompactionStateWithPlans(_, Property(&GetCompactionPlansRequest::compactionid, compaction_id), _))
        .WillOnce([](::grpc::ServerContext*, const GetCompactionPlansRequest*, GetCompactionPlansResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
        });

    ::milvus::CompactionPlans plans;
    auto status = client_->GetCompactionPlans(compaction_id, plans);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}