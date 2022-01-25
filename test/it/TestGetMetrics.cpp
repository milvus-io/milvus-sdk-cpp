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
using ::milvus::proto::milvus::GetMetricsRequest;
using ::milvus::proto::milvus::GetMetricsResponse;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, GetMetricsFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string request_str = "dummy_req";
    milvus::proto::milvus::GetMetricsRequest rpc_request;
    rpc_request.set_request(request_str);

    std::string response_str = "dummy_resp";
    std::string component_str = "dummy_comp";
    EXPECT_CALL(service_, GetMetrics(_, Property(&GetMetricsRequest::request, request_str), _))
        .WillOnce([response_str, component_str](::grpc::ServerContext*, const GetMetricsRequest* request,
                                                GetMetricsResponse* response) {
            response->set_response(response_str);
            response->set_component_name(component_str);
            return ::grpc::Status{};
        });

    std::string res_response, res_comp;
    auto status = client_->GetMetrics(request_str, res_response, res_comp);
    EXPECT_TRUE(status.IsOk());

    EXPECT_EQ(res_response, response_str);
    EXPECT_EQ(res_comp, component_str);
}

TEST_F(MilvusMockedTest, GetMetricsFooWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    std::string request_str = "dummy_req";
    milvus::proto::milvus::GetMetricsRequest rpc_request;
    rpc_request.set_request(request_str);

    std::string res_response, res_comp;
    auto status = client_->GetMetrics(request_str, res_response, res_comp);
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, GetMetricsFooFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string request_str = "dummy_req";
    milvus::proto::milvus::GetMetricsRequest rpc_request;
    rpc_request.set_request(request_str);

    EXPECT_CALL(service_, GetMetrics(_, Property(&GetMetricsRequest::request, request_str), _))
        .WillOnce([](::grpc::ServerContext*, const GetMetricsRequest*, GetMetricsResponse*) {
            return ::grpc::Status{::grpc::StatusCode::ABORTED, "Get metrics failed. Internal error"};
        });

    std::string res_response, res_comp;
    auto status = client_->GetMetrics(request_str, res_response, res_comp);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}