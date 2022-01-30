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
using ::milvus::proto::milvus::ReleaseCollectionRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, ReleaseCollectionFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection = "Foo";
    milvus::proto::milvus::ReleaseCollectionRequest request;
    request.set_collection_name(collection);

    EXPECT_CALL(service_, ReleaseCollection(_, Property(&ReleaseCollectionRequest::collection_name, collection), _))
        .WillOnce([](::grpc::ServerContext*, const ReleaseCollectionRequest* request,
                     ::milvus::proto::common::Status*) { return ::grpc::Status{}; });
    auto status = client_->ReleaseCollection(collection);
    EXPECT_TRUE(status.IsOk());
}