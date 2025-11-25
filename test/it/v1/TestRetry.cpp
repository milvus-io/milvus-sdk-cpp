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
#include "utils/GtsDict.h"

using ::milvus::StatusCode;
using ::milvus::proto::common::ErrorCode;
using ::milvus::proto::milvus::DescribeCollectionRequest;
using ::milvus::proto::milvus::DescribeCollectionResponse;

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, RetryMaxRetry) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);
    milvus::RetryParam retry_param;
    int max_retry_times = 3;
    client_->SetRetryParam(retry_param.WithMaxRetryTimes(max_retry_times));

    int call_times = 0;
    const std::string collection_name = "xxx";
    EXPECT_CALL(service_,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillRepeatedly([&call_times](::grpc::ServerContext*, const DescribeCollectionRequest* request,
                                      DescribeCollectionResponse* response) {
            response->mutable_status()->set_code(8);
            call_times++;
            return ::grpc::Status{};
        });

    ::milvus::CollectionDesc desc;
    auto status = client_->DescribeCollection(collection_name, desc);

    EXPECT_EQ(status.Code(), StatusCode::TIMEOUT);
    EXPECT_EQ(status.ServerCode(), 8);
    EXPECT_EQ(call_times, max_retry_times);
}

TEST_F(MilvusMockedTest, RetryRetryTimeout) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);
    milvus::RetryParam retry_param;
    int64_t max_retry_timeout_ms = 2000;
    client_->SetRetryParam(retry_param.WithMaxRetryTimeoutMs(max_retry_timeout_ms));

    const std::string collection_name = "xxx";
    EXPECT_CALL(service_,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillRepeatedly(
            [](::grpc::ServerContext*, const DescribeCollectionRequest* request, DescribeCollectionResponse* response) {
                response->mutable_status()->set_code(8);
                return ::grpc::Status{};
            });

    auto begin_ms = milvus::GetNowMs();
    ::milvus::CollectionDesc desc;
    auto status = client_->DescribeCollection(collection_name, desc);
    auto end_ms = milvus::GetNowMs();

    EXPECT_EQ(status.Code(), StatusCode::TIMEOUT);
    EXPECT_EQ(status.ServerCode(), 8);
    EXPECT_GE(end_ms - begin_ms, max_retry_timeout_ms);
}

TEST_F(MilvusMockedTest, RetrySuccess) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);
    milvus::RetryParam retry_param;
    client_->SetRetryParam(retry_param);

    int call_times = 0;
    const std::string collection_name = "xxx";
    EXPECT_CALL(service_,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillRepeatedly([&call_times](::grpc::ServerContext*, const DescribeCollectionRequest* request,
                                      DescribeCollectionResponse* response) {
            call_times++;
            response->mutable_status()->set_error_code((call_times < 3) ? ErrorCode::RateLimit : ErrorCode::Success);
            return ::grpc::Status{};
        });

    ::milvus::CollectionDesc desc;
    auto status = client_->DescribeCollection(collection_name, desc);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(call_times, 3);
}

TEST_F(MilvusMockedTest, RetryRpcErr) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);
    milvus::RetryParam retry_param;
    client_->SetRetryParam(retry_param);

    int call_times = 0;
    const std::string collection_name = "xxx";
    EXPECT_CALL(service_,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillRepeatedly([&call_times](::grpc::ServerContext*, const DescribeCollectionRequest* request,
                                      DescribeCollectionResponse* response) {
            response->mutable_status()->set_code(ErrorCode::Success);
            call_times++;
            return ::grpc::Status{::grpc::StatusCode::UNIMPLEMENTED, ""};
        });

    ::milvus::CollectionDesc desc;
    auto status = client_->DescribeCollection(collection_name, desc);

    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
    EXPECT_EQ(status.RpcErrCode(), static_cast<int32_t>(::grpc::StatusCode::UNIMPLEMENTED));
    EXPECT_EQ(call_times, 1);
}

TEST_F(MilvusMockedTest, RetryServerErr) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);
    milvus::RetryParam retry_param;
    client_->SetRetryParam(retry_param);

    int call_times = 0;
    const std::string collection_name = "xxx";
    EXPECT_CALL(service_,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillRepeatedly([&call_times](::grpc::ServerContext*, const DescribeCollectionRequest* request,
                                      DescribeCollectionResponse* response) {
            response->mutable_status()->set_code(10);  // code 10 is "server unimplemented"
            call_times++;
            return ::grpc::Status{};
        });

    ::milvus::CollectionDesc desc;
    auto status = client_->DescribeCollection(collection_name, desc);

    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
    EXPECT_EQ(status.ServerCode(), 10);
    EXPECT_EQ(call_times, 1);
}
