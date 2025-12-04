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

using ::milvus::StatusCode;
using ::milvus::proto::milvus::GetCompactionStateRequest;
using ::milvus::proto::milvus::GetCompactionStateResponse;
using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, GetCompactionState) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const int64_t compaction_id = 1;
    const int64_t executing_id = 100;
    const int64_t timeout_id = 101;
    const int64_t completed_id = 102;

    EXPECT_CALL(service_, GetCompactionState(_, Property(&GetCompactionStateRequest::compactionid, compaction_id), _))
        .WillOnce([&](::grpc::ServerContext*, const GetCompactionStateRequest*, GetCompactionStateResponse* response) {
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

TEST_F(UnconnectMilvusMockedTest, GetCompactionStateWithoutConnect) {
    const int64_t compaction_id = 1;
    ::milvus::CompactionState state;
    auto status = client_->GetCompactionState(compaction_id, state);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, GetCompactionStateFailed) {
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
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}
