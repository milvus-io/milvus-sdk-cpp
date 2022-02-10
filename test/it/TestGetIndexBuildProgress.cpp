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
using ::milvus::proto::milvus::GetIndexBuildProgressRequest;
using ::milvus::proto::milvus::GetIndexBuildProgressResponse;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Property;

TEST_F(MilvusMockedTest, GetIndexBuildProgressFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "Foo";
    std::string field_name = "Bar";
    milvus::IndexProgress index_progress;

    EXPECT_CALL(service_,
                GetIndexBuildProgress(_,
                                      AllOf(Property(&GetIndexBuildProgressRequest::collection_name, collection_name),
                                            Property(&GetIndexBuildProgressRequest::field_name, field_name)),
                                      _))
        .WillOnce(
            [](::grpc::ServerContext*, const GetIndexBuildProgressRequest*, GetIndexBuildProgressResponse* response) {
                response->set_total_rows(1000);
                response->set_indexed_rows(100);
                return grpc::Status{};
            });

    auto status = client_->GetIndexBuildProgress(collection_name, field_name, index_progress);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(index_progress.TotalRows(), 1000);
    EXPECT_EQ(index_progress.IndexedRows(), 100);
}