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

using ::milvus::StatusCode;
using ::milvus::proto::milvus::AlterCollectionSchemaRequest;
using ::milvus::proto::milvus::AlterCollectionSchemaResponse;
using ::testing::_;

namespace {

std::shared_ptr<milvus::MilvusClientV2>
CreateConnectedV2Client(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ::milvus::proto::milvus::ConnectRequest*,
                     ::milvus::proto::milvus::ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{"127.0.0.1", port};
    auto status = client->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());
    return client;
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldNotConnected) {
    auto client = milvus::MilvusClientV2::Create();
    auto status = client->DropCollectionField(
        milvus::DropCollectionFieldRequest().WithCollectionName("coll").WithFieldName("f1"));
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldServerFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse* response) {
                response->mutable_alter_status()->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
                response->mutable_alter_status()->set_reason("drop field failed");
                return ::grpc::Status{};
            });

    auto status = client->DropCollectionField(
        milvus::DropCollectionFieldRequest().WithCollectionName("coll").WithFieldName("f1"));
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldByName) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest* request,
                     AlterCollectionSchemaResponse* response) {
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_TRUE(request->action().has_drop_request());
            EXPECT_EQ(request->action().drop_request().field_name(), "f1");
            EXPECT_FALSE(request->action().drop_request().drop_function_output_fields());
            response->mutable_alter_status()->set_code(0);
            return ::grpc::Status{};
        });

    auto status = client->DropCollectionField(
        milvus::DropCollectionFieldRequest().WithCollectionName("coll").WithFieldName("f1"));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldById) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest* request,
                     AlterCollectionSchemaResponse* response) {
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_TRUE(request->action().has_drop_request());
            EXPECT_EQ(request->action().drop_request().field_id(), 101);
            response->mutable_alter_status()->set_code(0);
            return ::grpc::Status{};
        });

    auto status =
        client->DropCollectionField(milvus::DropCollectionFieldRequest().WithCollectionName("coll").WithFieldID(101));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldRejectsBothSelectors) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    auto status = client->DropCollectionField(
        milvus::DropCollectionFieldRequest().WithCollectionName("coll").WithFieldName("f1").WithFieldID(101));
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldRejectsMissingSelector) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    auto status = client->DropCollectionField(milvus::DropCollectionFieldRequest().WithCollectionName("coll"));
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldRejectsNegativeId) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    auto status =
        client->DropCollectionField(milvus::DropCollectionFieldRequest().WithCollectionName("coll").WithFieldID(-1));
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionFieldNotConnected) {
    auto client = milvus::MilvusClientV2::Create();
    milvus::FieldSchema field;
    field.SetName("sparse_vec");
    field.SetDataType(milvus::DataType::SPARSE_FLOAT_VECTOR);
    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("sparse_vec");

    auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                               .WithCollectionName("coll")
                                               .WithField(std::move(field))
                                               .WithFunction(function));
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionFieldServerFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field;
    field.SetName("sparse_vec");
    field.SetDataType(milvus::DataType::SPARSE_FLOAT_VECTOR);
    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("sparse_vec");

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse* response) {
                response->mutable_alter_status()->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
                response->mutable_alter_status()->set_reason("add function field failed");
                return ::grpc::Status{};
            });

    auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                               .WithCollectionName("coll")
                                               .WithField(std::move(field))
                                               .WithFunction(function));
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionFieldRejectsMismatchedOutputName) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field;
    field.SetName("sparse_vec");
    field.SetDataType(milvus::DataType::SPARSE_FLOAT_VECTOR);
    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("other_vec");

    auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                               .WithCollectionName("coll")
                                               .WithField(std::move(field))
                                               .WithFunction(function));
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionField) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field;
    field.SetName("sparse_vec");
    field.SetDataType(milvus::DataType::SPARSE_FLOAT_VECTOR);
    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("sparse_vec");

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest* request,
                     AlterCollectionSchemaResponse* response) {
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_TRUE(request->action().has_add_request());
            EXPECT_EQ(request->action().add_request().field_infos_size(), 1);
            if (request->action().add_request().field_infos_size() != 1) {
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->action().add_request().field_infos(0).field_schema().name(), "sparse_vec");
            EXPECT_TRUE(request->action().add_request().field_infos(0).field_schema().is_function_output());
            EXPECT_EQ(request->action().add_request().func_schema_size(), 1);
            if (request->action().add_request().func_schema_size() != 1) {
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->action().add_request().func_schema(0).name(), "bm25_fn");
            response->mutable_alter_status()->set_code(0);
            return ::grpc::Status{};
        });

    auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                               .WithCollectionName("coll")
                                               .WithField(std::move(field))
                                               .WithFunction(function));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, DropFunctionFieldNotConnected) {
    auto client = milvus::MilvusClientV2::Create();
    auto status = client->DropFunctionField(
        milvus::DropFunctionFieldRequest().WithCollectionName("coll").WithFunctionName("bm25_fn"));
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(UnconnectMilvusMockedTest, DropFunctionFieldServerFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse* response) {
                response->mutable_alter_status()->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
                response->mutable_alter_status()->set_reason("drop function field failed");
                return ::grpc::Status{};
            });

    auto status = client->DropFunctionField(
        milvus::DropFunctionFieldRequest().WithCollectionName("coll").WithFunctionName("bm25_fn"));
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, DropFunctionField) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest* request,
                     AlterCollectionSchemaResponse* response) {
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_TRUE(request->action().has_drop_request());
            EXPECT_EQ(request->action().drop_request().function_name(), "bm25_fn");
            EXPECT_TRUE(request->action().drop_request().drop_function_output_fields());
            response->mutable_alter_status()->set_code(0);
            return ::grpc::Status{};
        });

    auto status = client->DropFunctionField(
        milvus::DropFunctionFieldRequest().WithCollectionName("coll").WithFunctionName("bm25_fn"));
    EXPECT_TRUE(status.IsOk());
}
