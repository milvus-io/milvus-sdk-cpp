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

#include "TypeUtils.h"
#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::CreateIndexRequest;
using ::milvus::proto::milvus::GetIndexStateRequest;
using ::milvus::proto::milvus::GetIndexStateResponse;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Property;

TEST_F(MilvusMockedTest, TestCreateIndexInstantly) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "test_collection";
    std::string field_name = "test_field";
    std::string index_name = "test_index";
    auto index_type = milvus::IndexType::IVF_FLAT;
    auto metric_type = milvus::MetricType::L2;
    int64_t index_id = 0;

    milvus::IndexDesc index_desc(field_name, "", index_type, metric_type, index_id);
    index_desc.AddExtraParam("nlist", 1024);
    const auto progress_monitor = ::milvus::ProgressMonitor::NoWait();
    EXPECT_CALL(service_, CreateIndex(_,
                                      AllOf(Property(&CreateIndexRequest::collection_name, collection_name),
                                            Property(&CreateIndexRequest::field_name, field_name)),
                                      _))
        .WillOnce([](::grpc::ServerContext*, const CreateIndexRequest*, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });
    auto status = client_->CreateIndex(collection_name, index_desc, progress_monitor);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, TestCreateIndexWithProgress) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "test_collection";
    std::string field_name = "test_field";
    auto index_type = milvus::IndexType::IVF_FLAT;
    auto metric_type = milvus::MetricType::L2;
    int64_t index_id = 0;

    milvus::IndexDesc index_desc(field_name, "", index_type, metric_type, index_id);
    index_desc.AddExtraParam("nlist", 1024);
    auto progress_monitor = ::milvus::ProgressMonitor::Forever();
    progress_monitor.SetCheckInterval(10);
    EXPECT_CALL(service_, CreateIndex(_,
                                      AllOf(Property(&CreateIndexRequest::collection_name, collection_name),
                                            Property(&CreateIndexRequest::field_name, field_name)),
                                      _))
        .WillOnce([index_type, metric_type](::grpc::ServerContext*, const CreateIndexRequest* req,
                                            ::milvus::proto::common::Status* status) {
            std::unordered_map<std::string, std::string> params{};
            for (const auto& pair : req->extra_params()) {
                params.emplace(pair.key(), pair.value());
            }
            EXPECT_EQ(params[milvus::KeyIndexType()], std::to_string(index_type));
            EXPECT_EQ(params[milvus::KeyMetricType()], std::to_string(metric_type));
            EXPECT_EQ(params[milvus::KeyParams()], "{\"nlist\":1024}");

            status->set_error_code(milvus::proto::common::ErrorCode::Success);
            return ::grpc::Status{};
        });

    int called_times{0};
    EXPECT_CALL(service_, GetIndexState(_,
                                        AllOf(Property(&GetIndexStateRequest::collection_name, collection_name),
                                              Property(&GetIndexStateRequest::field_name, field_name)),
                                        _))
        .Times(10)
        .WillRepeatedly(
            [&](::grpc::ServerContext*, const GetIndexStateRequest* request, GetIndexStateResponse* response) {
                called_times++;
                milvus::proto::common::IndexState state = (called_times == 10)
                                                              ? milvus::proto::common::IndexState::Finished
                                                              : milvus::proto::common::IndexState::InProgress;
                response->set_state(state);
                return ::grpc::Status{};
            });

    auto status = client_->CreateIndex(collection_name, index_desc, progress_monitor);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, TestCreateIndexFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "test_collection";
    std::string field_name = "test_field";
    auto index_type = milvus::IndexType::IVF_FLAT;
    auto metric_type = milvus::MetricType::L2;
    int64_t index_id = 0;

    milvus::IndexDesc index_desc(field_name, "", index_type, metric_type, index_id);
    index_desc.AddExtraParam("nlist", 1024);
    auto progress_monitor = ::milvus::ProgressMonitor::Forever();
    progress_monitor.SetCheckInterval(10);
    EXPECT_CALL(service_, CreateIndex(_,
                                      AllOf(Property(&CreateIndexRequest::collection_name, collection_name),
                                            Property(&CreateIndexRequest::field_name, field_name)),
                                      _))
        .WillRepeatedly([](::grpc::ServerContext*, const CreateIndexRequest*, ::milvus::proto::common::Status* status) {
            status->set_error_code(milvus::proto::common::ErrorCode::Success);
            return ::grpc::Status{};
        });

    std::string failed_reason = "unknow";
    EXPECT_CALL(service_, GetIndexState(_, _, _))
        .WillOnce([&failed_reason](::grpc::ServerContext*, const GetIndexStateRequest*, GetIndexStateResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, failed_reason};
        });

    auto status = client_->CreateIndex(collection_name, index_desc, progress_monitor);
    EXPECT_FALSE(status.IsOk());

    EXPECT_CALL(service_, GetIndexState(_, _, _))
        .WillOnce(
            [&failed_reason](::grpc::ServerContext*, const GetIndexStateRequest*, GetIndexStateResponse* response) {
                response->set_state(milvus::proto::common::IndexState::Failed);
                response->set_fail_reason(failed_reason);
                return ::grpc::Status{};
            });

    status = client_->CreateIndex(collection_name, index_desc, progress_monitor);
    EXPECT_FALSE(status.IsOk());
}