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
#include "utils/CompareUtils.h"
#include "utils/TypeUtils.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::CreateIndexRequest;
using ::milvus::proto::milvus::DescribeIndexRequest;
using ::milvus::proto::milvus::DescribeIndexResponse;
using ::milvus::proto::milvus::FlushRequest;
using ::milvus::proto::milvus::FlushResponse;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Property;

TEST_F(MilvusMockedTest, CreateIndexInstantly) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "test_collection";
    std::string field_name = "test_field";
    std::string index_name = "test_index";
    auto index_type = milvus::IndexType::IVF_FLAT;
    auto metric_type = milvus::MetricType::L2;

    milvus::IndexDesc index_desc(field_name, "", index_type, metric_type);
    index_desc.AddExtraParam("nlist", "1024");
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

TEST_F(MilvusMockedTest, CreateIndexWithProgress) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "test_collection";
    std::string field_name = "test_field";
    auto index_type = milvus::IndexType::IVF_FLAT;
    auto metric_type = milvus::MetricType::L2;

    milvus::IndexDesc index_desc(field_name, "", index_type, metric_type);
    index_desc.AddExtraParam("nlist", "1024");
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
            EXPECT_EQ(params[milvus::INDEX_TYPE], std::to_string(index_type));
            EXPECT_EQ(params[milvus::METRIC_TYPE], std::to_string(metric_type));

            status->set_code(milvus::proto::common::ErrorCode::Success);
            return ::grpc::Status{};
        });

    int called_times{0};
    EXPECT_CALL(service_, DescribeIndex(_,
                                        AllOf(Property(&DescribeIndexRequest::collection_name, collection_name),
                                              Property(&DescribeIndexRequest::field_name, field_name)),
                                        _))
        .Times(10)
        .WillRepeatedly(
            [&](::grpc::ServerContext*, const DescribeIndexRequest* request, DescribeIndexResponse* response) {
                called_times++;
                milvus::proto::common::IndexState state = (called_times == 10)
                                                              ? milvus::proto::common::IndexState::Finished
                                                              : milvus::proto::common::IndexState::InProgress;
                milvus::proto::milvus::IndexDescription* desc = response->add_index_descriptions();
                desc->set_field_name(field_name);
                desc->set_state(state);
                return ::grpc::Status{};
            });

    auto status = client_->CreateIndex(collection_name, index_desc, progress_monitor);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, CreateIndexFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "test_collection";
    std::string field_name = "test_field";
    auto index_type = milvus::IndexType::IVF_FLAT;
    auto metric_type = milvus::MetricType::L2;

    milvus::IndexDesc index_desc(field_name, "", index_type, metric_type);
    index_desc.AddExtraParam("nlist", "1024");
    auto progress_monitor = ::milvus::ProgressMonitor::Forever();
    progress_monitor.SetCheckInterval(10);

    EXPECT_CALL(service_, Flush(_, AllOf(Property(&FlushRequest::collection_names, ElementsAre(collection_name))), _))
        .WillRepeatedly([&](::grpc::ServerContext*, const FlushRequest*, FlushResponse*) { return ::grpc::Status{}; });

    EXPECT_CALL(service_, CreateIndex(_,
                                      AllOf(Property(&CreateIndexRequest::collection_name, collection_name),
                                            Property(&CreateIndexRequest::field_name, field_name)),
                                      _))
        .WillRepeatedly([](::grpc::ServerContext*, const CreateIndexRequest*, ::milvus::proto::common::Status* status) {
            status->set_code(milvus::proto::common::ErrorCode::Success);
            return ::grpc::Status{};
        });

    std::string failed_reason = "unknow";
    EXPECT_CALL(service_, DescribeIndex(_, _, _))
        .WillOnce([&failed_reason](::grpc::ServerContext*, const DescribeIndexRequest*, DescribeIndexResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, failed_reason};
        });

    auto status = client_->CreateIndex(collection_name, index_desc, progress_monitor);
    EXPECT_FALSE(status.IsOk());

    EXPECT_CALL(service_, DescribeIndex(_, _, _))
        .WillOnce([&failed_reason, &field_name](::grpc::ServerContext*, const DescribeIndexRequest*,
                                                DescribeIndexResponse* response) {
            milvus::proto::milvus::IndexDescription* desc = response->add_index_descriptions();
            desc->set_field_name(field_name);
            desc->set_state(milvus::proto::common::IndexState::Failed);
            desc->set_index_state_fail_reason(failed_reason);
            return ::grpc::Status{};
        });

    status = client_->CreateIndex(collection_name, index_desc, progress_monitor);
    EXPECT_FALSE(status.IsOk());
}