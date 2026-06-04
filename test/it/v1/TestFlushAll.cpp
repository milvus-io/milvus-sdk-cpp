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
#include "milvus/MilvusClientV2.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::FlushAllRequest;
using ::milvus::proto::milvus::FlushAllResponse;
using ::milvus::proto::milvus::GetFlushAllStateRequest;
using ::milvus::proto::milvus::GetFlushAllStateResponse;
using ::testing::_;
using ::testing::Property;

namespace {

milvus::MilvusClientV2Ptr
CreateConnectedV2Client(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{"127.0.0.1", port};
    auto status = client->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());
    return client;
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, FlushAll) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, FlushAll(_, Property(&FlushAllRequest::db_name, "db1"), _))
        .WillOnce([](::grpc::ServerContext*, const FlushAllRequest*, FlushAllResponse* response) {
            response->set_flush_all_ts(12345);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, GetFlushAllState(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const GetFlushAllStateRequest* request, GetFlushAllStateResponse* response) {
                EXPECT_EQ(request->db_name(), "db1");
                EXPECT_EQ(request->flush_all_ts(), 12345);
                response->set_flushed(true);
                return ::grpc::Status{};
            });

    milvus::FlushAllResponse response;
    auto status = client->FlushAll(milvus::FlushAllRequest().WithDatabaseName("db1"), response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.FlushAllTs(), 12345);
}

TEST_F(UnconnectMilvusMockedTest, FlushAllFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, FlushAll(_, Property(&FlushAllRequest::db_name, "db1"), _))
        .WillOnce([](::grpc::ServerContext*, const FlushAllRequest*, FlushAllResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::FlushAllResponse response;
    auto status = client->FlushAll(milvus::FlushAllRequest().WithDatabaseName("db1"), response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, FlushAllServerFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, FlushAll(_, Property(&FlushAllRequest::db_name, "db1"), _))
        .WillOnce([](::grpc::ServerContext*, const FlushAllRequest*, FlushAllResponse* response) {
            response->mutable_status()->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
            return ::grpc::Status{};
        });

    milvus::FlushAllResponse response;
    auto status = client->FlushAll(milvus::FlushAllRequest().WithDatabaseName("db1"), response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, GetFlushAllState) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetFlushAllState(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const GetFlushAllStateRequest* request, GetFlushAllStateResponse* response) {
                EXPECT_EQ(request->db_name(), "db1");
                EXPECT_EQ(request->flush_all_ts(), 12345);
                response->set_flushed(true);
                return ::grpc::Status{};
            });

    milvus::GetFlushAllStateResponse response;
    auto status = client->GetFlushAllState(
        milvus::GetFlushAllStateRequest().WithDatabaseName("db1").WithFlushAllTs(12345), response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_TRUE(response.Flushed());
}

TEST_F(UnconnectMilvusMockedTest, GetFlushAllStateFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetFlushAllState(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetFlushAllStateRequest*, GetFlushAllStateResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::GetFlushAllStateResponse response;
    auto status = client->GetFlushAllState(
        milvus::GetFlushAllStateRequest().WithDatabaseName("db1").WithFlushAllTs(12345), response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, GetFlushAllStateServerFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetFlushAllState(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetFlushAllStateRequest*, GetFlushAllStateResponse* response) {
            response->mutable_status()->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
            return ::grpc::Status{};
        });

    milvus::GetFlushAllStateResponse response;
    auto status = client->GetFlushAllState(
        milvus::GetFlushAllStateRequest().WithDatabaseName("db1").WithFlushAllTs(12345), response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}
