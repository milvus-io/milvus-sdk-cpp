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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::common::IndexState;
using ::milvus::proto::milvus::GetIndexStateRequest;
using ::milvus::proto::milvus::GetIndexStateResponse;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Property;

TEST_F(MilvusMockedTest, GetIndexStateFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "Foo";
    std::string field_name = "Bar";
    milvus::IndexState index_state;

    EXPECT_CALL(service_, GetIndexState(_,
                                        AllOf(Property(&GetIndexStateRequest::collection_name, collection_name),
                                              Property(&GetIndexStateRequest::field_name, field_name)),
                                        _))
        .WillOnce([](::grpc::ServerContext*, const GetIndexStateRequest*, GetIndexStateResponse* response) {
            response->set_state(IndexState::Finished);
            return grpc::Status{};
        });

    auto status = client_->GetIndexState(collection_name, field_name, index_state);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(index_state.StateCode(), milvus::IndexStateCode::FINISHED);
}