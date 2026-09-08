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

#include <stdexcept>

#include "../mocks/MilvusMockedTest.h"
#include "utils/ConnectionHandler.h"
#include "milvus/types/ConnectParam.h"

using ::milvus::proto::milvus::CheckHealthRequest;
using ::milvus::proto::milvus::CheckHealthResponse;
using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::testing::_;

namespace {

milvus::Status
ConnectHandler(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port,
               milvus::ConnectionHandler& handler) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) { return ::grpc::Status{}; });
    return handler.Connect(milvus::ConnectParam{"127.0.0.1", port});
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, ExceptionBarrierConvertsStdExceptionToStatus) {
    milvus::ConnectionHandler handler;
    auto connect_status = ConnectHandler(service_, server_.ListenPort(), handler);
    ASSERT_TRUE(connect_status.IsOk());

    auto throwing_validate = []() -> milvus::Status { throw std::runtime_error("boom"); };
    auto pre = [](CheckHealthRequest&) { return milvus::Status::OK(); };
    auto post = [](const CheckHealthResponse&) { return milvus::Status::OK(); };

    auto status = handler.Invoke<CheckHealthRequest, CheckHealthResponse>(
        throwing_validate, pre, &milvus::MilvusConnection::CheckHealth, post);

    EXPECT_EQ(status.Code(), milvus::StatusCode::UNKNOWN_ERROR);
    EXPECT_NE(status.Message().find("Unexpected SDK exception: boom"), std::string::npos);
}

TEST_F(UnconnectMilvusMockedTest, ExceptionBarrierConvertsNonStdExceptionToStatus) {
    milvus::ConnectionHandler handler;
    auto connect_status = ConnectHandler(service_, server_.ListenPort(), handler);
    ASSERT_TRUE(connect_status.IsOk());

    auto throwing_validate = []() -> milvus::Status { throw 42; };
    auto pre = [](CheckHealthRequest&) { return milvus::Status::OK(); };
    auto post = [](const CheckHealthResponse&) { return milvus::Status::OK(); };

    auto status = handler.Invoke<CheckHealthRequest, CheckHealthResponse>(
        throwing_validate, pre, &milvus::MilvusConnection::CheckHealth, post);

    EXPECT_EQ(status.Code(), milvus::StatusCode::UNKNOWN_ERROR);
    EXPECT_NE(status.Message().find("Unexpected SDK exception"), std::string::npos);
}
