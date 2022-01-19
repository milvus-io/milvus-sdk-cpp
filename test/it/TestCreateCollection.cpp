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
using ::milvus::proto::milvus::CreateCollectionRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, CreateCollectionFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    milvus::CollectionSchema collection_schema("Foo");
    collection_schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
    collection_schema.AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
    collection_schema.AddField(
        milvus::FieldSchema("face", milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(4));

    milvus::proto::milvus::CreateCollectionRequest rpc_request;
    rpc_request.set_collection_name("Foo");

    EXPECT_CALL(service_, CreateCollection(_, Property(&CreateCollectionRequest::collection_name, "Foo"), _))
        .WillOnce([](::grpc::ServerContext*, const CreateCollectionRequest* request, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });
    auto status = client_->CreateCollection(collection_schema);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, CreateCollectionFooWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    milvus::CollectionSchema collection_schema("Foo");
    milvus::proto::milvus::CreateCollectionRequest rpc_request;
    rpc_request.set_collection_name("Foo");

    auto status = client_->CreateCollection(collection_schema);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, CreateCollectionFooFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    milvus::CollectionSchema collection_schema("Foo");
    milvus::proto::milvus::CreateCollectionRequest rpc_request;
    rpc_request.set_collection_name("Foo");

    auto error_code = milvus::proto::common::ErrorCode::UnexpectedError;
    EXPECT_CALL(service_, CreateCollection(_, Property(&CreateCollectionRequest::collection_name, "Foo"), _))
        .WillOnce([error_code](::grpc::ServerContext*, const CreateCollectionRequest* request,
                               ::milvus::proto::common::Status* status) {
            status->set_error_code(error_code);
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
        });
    auto status = client_->CreateCollection(collection_schema);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}
