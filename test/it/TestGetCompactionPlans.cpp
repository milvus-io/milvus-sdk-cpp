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
using ::milvus::proto::milvus::GetCompactionPlansRequest;
using ::milvus::proto::milvus::GetCompactionPlansResponse;
using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, GetCompactionPlans) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const int64_t compaction_id = 1;
    const std::vector<int64_t> sources = {1, 2, 3, 4};
    const int64_t target = 100;

    EXPECT_CALL(service_,
                GetCompactionStateWithPlans(_, Property(&GetCompactionPlansRequest::compactionid, compaction_id), _))
        .WillOnce([&](::grpc::ServerContext*, const GetCompactionPlansRequest*, GetCompactionPlansResponse* response) {
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

TEST_F(UnconnectMilvusMockedTest, GetCompactionPlansConnect) {
    const int64_t compaction_id = 1;
    ::milvus::CompactionPlans plans;
    auto status = client_->GetCompactionPlans(compaction_id, plans);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, GetCompactionPlansFailed) {
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
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}
