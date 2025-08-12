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

#include <thread>

#include "mocks/MilvusMockedTest.h"

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

TEST_F(UnconnectMilvusMockedTest, ConnectFailed) {
    auto port = server_.ListenPort();
    milvus::ConnectParam connect_param{"127.0.0.1", ++port};
    auto status = client_->Connect(connect_param);
    EXPECT_FALSE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, ConnectTimeout) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    connect_param.SetConnectTimeout(10);
    auto status = DoConnect(service_, client_, connect_param, 100);
    EXPECT_FALSE(status.IsOk());
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