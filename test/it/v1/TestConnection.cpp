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

#include <condition_variable>
#include <mutex>
#include <thread>

#include "../mocks/MilvusMockedTest.h"
#include "utils/ConnectionHandler.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::HasCollectionRequest;

using ::testing::_;
using ::testing::AllOf;
using ::testing::Property;

milvus::Status
DoConnect(testing::StrictMock<::milvus::MilvusMockedService>& service_,
          std::shared_ptr<::milvus::MilvusClient>& client_, const milvus::ConnectParam& param,
          uint64_t simulate_timeout = 0) {
    EXPECT_CALL(service_, Connect(_, _, _))
        .WillOnce([&simulate_timeout](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse* response) {
            // sleep if timeout
            if (simulate_timeout > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds{simulate_timeout});
            }
            return ::grpc::Status{};
        });

    return client_->Connect(param);
}

TEST_F(UnconnectMilvusMockedTest, ConnectSuccessful) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    connect_param.SetConnectTimeout(100);
    auto status = DoConnect(service_, client_, connect_param, 10);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, ResolveDatabaseNamePreservesEmptyForRpc) {
    EXPECT_CALL(service_, Connect(_, _, _))
        .Times(2)
        .WillRepeatedly(
            [](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) { return ::grpc::Status{}; });

    milvus::ConnectionHandler empty_db_handler;
    ASSERT_TRUE(empty_db_handler.Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    EXPECT_EQ(empty_db_handler.CurrentDbName(""), "");
    EXPECT_EQ(empty_db_handler.CurrentDbName("request_db"), "request_db");

    milvus::ConnectParam explicit_db_param{"127.0.0.1", server_.ListenPort()};
    explicit_db_param.SetDbName("connected_db");
    milvus::ConnectionHandler explicit_db_handler;
    ASSERT_TRUE(explicit_db_handler.Connect(explicit_db_param).IsOk());
    EXPECT_EQ(explicit_db_handler.CurrentDbName(""), "connected_db");
    EXPECT_EQ(explicit_db_handler.CurrentDbName("request_db"), "request_db");
}

TEST_F(UnconnectMilvusMockedTest, ConnectServerRejected) {
    EXPECT_CALL(service_, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse* response) {
            response->mutable_status()->set_code(1);
            response->mutable_status()->set_reason("server rejected connection");
            return ::grpc::Status{};
        });

    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    auto status = client_->Connect(connect_param);
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
    EXPECT_EQ(status.Message(), "server rejected connection");

    bool has_collection = false;
    status = client_->HasCollection("collection", has_collection);
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(UnconnectMilvusMockedTest, ConnectFailed) {
    auto port = server_.ListenPort();
    milvus::ConnectParam connect_param{"127.0.0.1", ++port};
    connect_param.SetConnectTimeout(10);
    auto status = client_->Connect(connect_param);
    EXPECT_FALSE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, ConnectTimeout) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    connect_param.SetConnectTimeout(10);
    auto status = DoConnect(service_, client_, connect_param, 100);
    EXPECT_FALSE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, FailedReconnectPreservesExistingConnection) {
    milvus::ConnectParam initial_connect_param{"127.0.0.1", server_.ListenPort(), "token-a"};
    auto status = DoConnect(service_, client_, initial_connect_param);
    ASSERT_TRUE(status.IsOk());

    milvus::ConnectParam reconnect_param{"127.0.0.1", server_.ListenPort(), "token-b"};
    const auto reconnect_authorization = reconnect_param.Authorizations();
    EXPECT_CALL(service_, Connect(_, _, _))
        .WillOnce([reconnect_authorization](::grpc::ServerContext* context, const ConnectRequest*,
                                            ConnectResponse* response) {
            const auto& metadata = context->client_metadata();
            auto authorization = metadata.find("authorization");
            EXPECT_NE(authorization, metadata.end());
            EXPECT_EQ(authorization->second, reconnect_authorization);
            response->mutable_status()->set_code(1);
            response->mutable_status()->set_reason("server rejected reconnection");
            return ::grpc::Status{};
        });
    status = client_->Connect(reconnect_param);
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);

    const auto initial_authorization = initial_connect_param.Authorizations();
    EXPECT_CALL(service_, HasCollection(_, Property(&HasCollectionRequest::collection_name, "collection"), _))
        .WillOnce([initial_authorization](::grpc::ServerContext* context, const HasCollectionRequest*,
                                          ::milvus::proto::milvus::BoolResponse* response) {
            const auto& metadata = context->client_metadata();
            auto authorization = metadata.find("authorization");
            EXPECT_NE(authorization, metadata.end());
            EXPECT_EQ(authorization->second, initial_authorization);
            response->set_value(true);
            return ::grpc::Status{};
        });
    bool has_collection = false;
    status = client_->HasCollection("collection", has_collection);
    EXPECT_TRUE(status.IsOk());
    EXPECT_TRUE(has_collection);
}

TEST_F(UnconnectMilvusMockedTest, ConnectionMutationWaitsForConnectAndAppliesToNewConnection) {
    milvus::ConnectionHandler handler;
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    EXPECT_CALL(service_, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) { return ::grpc::Status{}; });
    ASSERT_TRUE(handler.Connect(connect_param).IsOk());

    std::mutex handshake_mutex;
    std::condition_variable handshake_cv;
    bool handshake_started = false;
    bool finish_handshake = false;
    EXPECT_CALL(service_, Connect(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) {
            std::unique_lock<std::mutex> lock(handshake_mutex);
            handshake_started = true;
            handshake_cv.notify_all();
            handshake_cv.wait(lock, [&] { return finish_handshake; });
            return ::grpc::Status{};
        });

    milvus::Status connect_status;
    std::thread connect_thread([&] { connect_status = handler.Connect(connect_param); });
    {
        std::unique_lock<std::mutex> lock(handshake_mutex);
        handshake_cv.wait(lock, [&] { return handshake_started; });
    }

    std::mutex setter_mutex;
    std::condition_variable setter_cv;
    bool setter_started = false;
    bool setter_finished = false;
    milvus::Status setter_status;
    std::thread setter_thread([&] {
        {
            std::lock_guard<std::mutex> lock(setter_mutex);
            setter_started = true;
        }
        setter_cv.notify_all();
        setter_status = handler.SetRpcDeadlineMs(1234);
        {
            std::lock_guard<std::mutex> lock(setter_mutex);
            setter_finished = true;
        }
        setter_cv.notify_all();
    });

    {
        std::unique_lock<std::mutex> lock(setter_mutex);
        setter_cv.wait(lock, [&] { return setter_started; });
        EXPECT_FALSE(setter_cv.wait_for(lock, std::chrono::milliseconds(50), [&] { return setter_finished; }));
    }

    {
        std::lock_guard<std::mutex> lock(handshake_mutex);
        finish_handshake = true;
    }
    handshake_cv.notify_all();

    connect_thread.join();
    setter_thread.join();
    EXPECT_TRUE(connect_status.IsOk());
    EXPECT_TRUE(setter_status.IsOk());
    EXPECT_EQ(handler.GetRpcDeadlineMs(), 1234);
}

TEST_F(UnconnectMilvusMockedTest, ConnectWithUsername) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort(), "username", "password"};
    auto status = DoConnect(service_, client_, connect_param, 10);

    EXPECT_EQ(connect_param.Authorizations(), "dXNlcm5hbWU6cGFzc3dvcmQ=");

    std::string collection_name = "Foo";

    EXPECT_CALL(service_, HasCollection(_, Property(&HasCollectionRequest::collection_name, collection_name), _))
        .WillOnce([](::grpc::ServerContext* context, const HasCollectionRequest*,
                     ::milvus::proto::milvus::BoolResponse* response) {
            // check context
            auto& meta = context->client_metadata();
            auto it = meta.find("authorization");
            EXPECT_NE(it, meta.end());
            EXPECT_EQ(it->second, "dXNlcm5hbWU6cGFzc3dvcmQ=");

            response->set_value(false);
            return ::grpc::Status{};
        });
    bool has_collection{false};
    status = client_->HasCollection(collection_name, has_collection);

    EXPECT_TRUE(status.IsOk());
    EXPECT_FALSE(has_collection);
}
