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

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::DeleteRequest;
using ::milvus::proto::milvus::MutationResult;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, DeleteFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::vector<int64_t> ret_ids{1000, 10001, 1002, 1003};

    EXPECT_CALL(service_, Delete(_,
                                 AllOf(Property(&DeleteRequest::collection_name, "collection"),
                                       Property(&DeleteRequest::partition_name, "partition"),
                                       Property(&DeleteRequest::expr, "dummy expr")),
                                 _))
        .WillOnce([&ret_ids](::grpc::ServerContext*, const DeleteRequest* request, MutationResult* response) {
            // ret
            response->mutable_ids()->mutable_int_id()->mutable_data()->Add(ret_ids.begin(), ret_ids.end());
            return ::grpc::Status{};
        });

    milvus::DmlResults results;
    auto status = client_->Delete("collection", "partition", "dummy expr", results);

    EXPECT_TRUE(status.IsOk());
    EXPECT_TRUE(results.IdArray().IsIntegerID());
    EXPECT_THAT(results.IdArray().IntIDArray(), ElementsAreArray(ret_ids));
}

TEST_F(MilvusMockedTest, DeleteWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    milvus::DmlResults results;
    auto status = client_->Delete("collection", "partition", "dummy expr", results);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}
