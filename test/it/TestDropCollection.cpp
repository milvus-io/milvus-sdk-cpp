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
using ::milvus::proto::milvus::DropCollectionRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, DropCollectionFooSuccess) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "Foo";
    milvus::proto::milvus::DropCollectionRequest rpc_request;
    rpc_request.set_collection_name(collection_name);

    EXPECT_CALL(service_, DropCollection(_, Property(&DropCollectionRequest::collection_name, collection_name), _))
        .WillOnce([](::grpc::ServerContext*, const DropCollectionRequest* request, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });
    auto status = client_->DropCollection(collection_name);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, DropCollectionFooWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    std::string collection_name = "Foo";
    milvus::proto::milvus::DropCollectionRequest rpc_request;
    rpc_request.set_collection_name(collection_name);

    auto status = client_->DropCollection(collection_name);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, DropCollectionFooFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "Foo";
    milvus::proto::milvus::DropCollectionRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    auto error_code = milvus::proto::common::ErrorCode::UnexpectedError;

    EXPECT_CALL(service_, DropCollection(_, Property(&DropCollectionRequest::collection_name, collection_name), _))
        .WillOnce([error_code](::grpc::ServerContext*, const DropCollectionRequest*,
                               ::milvus::proto::common::Status* status) {
            status->set_error_code(error_code);
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, "Collection drop failed. Internal error"};
        });
    auto status = client_->DropCollection(collection_name);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}