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

#include <chrono>

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::FlushRequest;
using ::milvus::proto::milvus::FlushResponse;
using ::milvus::proto::milvus::GetFlushStateRequest;
using ::milvus::proto::milvus::GetFlushStateResponse;
using ::milvus::proto::schema::LongArray;
using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, FlushWithoutConnect) {
    const std::vector<std::string> collections{"c1", "c2"};
    const auto progress_monitor = ::milvus::ProgressMonitor::NoWait();
    auto status = client_->Flush(collections, progress_monitor);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, FlushInstantly) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::vector<std::string> collections{"c1", "c2"};
    const auto progress_monitor = ::milvus::ProgressMonitor::NoWait();

    EXPECT_CALL(service_, Flush(_, AllOf(Property(&FlushRequest::collection_names_size, collections.size())), _))
        .WillOnce(
            [](::grpc::ServerContext*, const FlushRequest*, FlushResponse* response) { return ::grpc::Status{}; });

    auto status = client_->Flush(collections, progress_monitor);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, FlushFailure) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::vector<std::string> collections{"c1", "c2"};
    const auto progress_monitor = ::milvus::ProgressMonitor::NoWait();

    EXPECT_CALL(service_, Flush(_, AllOf(Property(&FlushRequest::collection_names_size, collections.size())), _))
        .WillOnce([](::grpc::ServerContext*, const FlushRequest*, FlushResponse* response) {
            response->mutable_status()->set_error_code(::milvus::proto::common::ErrorCode::UnexpectedError);
            return ::grpc::Status{};
        });

    auto status = client_->Flush(collections, progress_monitor);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(MilvusMockedTest, FlushSuccess) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::vector<std::string> collections{"c1", "c2"};
    ::milvus::ProgressMonitor progress_monitor{1};
    std::vector<milvus::Progress> progresses{};
    progress_monitor.SetCheckInterval(1);
    progress_monitor.SetCallbackFunc([&progresses](milvus::Progress& progress) { progresses.emplace_back(progress); });

    const std::vector<int64_t> c1_ids{1, 2};
    const std::vector<int64_t> c2_ids{101, 102, 103};

    EXPECT_CALL(service_, Flush(_, AllOf(Property(&FlushRequest::collection_names_size, collections.size())), _))
        .WillOnce([&](::grpc::ServerContext*, const FlushRequest*, FlushResponse* response) {
            LongArray c1_segids, c2_segids;
            for (auto id : c1_ids) {
                c1_segids.add_data(id);
            }
            for (auto id : c2_ids) {
                c2_segids.add_data(id);
            }

            auto segids = response->mutable_coll_segids();
            (*segids)[collections[0]] = c1_segids;
            (*segids)[collections[1]] = c2_segids;
            return ::grpc::Status{};
        });

    int get_flush_state_called = 0;
    EXPECT_CALL(service_, GetFlushState(_, _, _))
        .Times(3)
        .WillRepeatedly([&c1_ids, &c2_ids, &get_flush_state_called](::grpc::ServerContext*,
                                                                    const GetFlushStateRequest* request,
                                                                    GetFlushStateResponse* response) {
            ++get_flush_state_called;
            auto ids = request->segmentids();
            std::vector<int64_t> seg_ids;
            seg_ids.insert(seg_ids.end(), ids.begin(), ids.end());

            if (seg_ids.size() == c1_ids.size()) {
                EXPECT_THAT(seg_ids, ElementsAreArray(c1_ids));
            } else if (seg_ids.size() == c2_ids.size()) {
                EXPECT_THAT(seg_ids, ElementsAreArray(c2_ids));
            }

            // first call set c1 to flushed
            // second call set c2 to unflushed
            // third call set c2 to flushed
            response->set_flushed(get_flush_state_called % 2 == 1);

            return ::grpc::Status{};
        });

    uint32_t segment_count = c1_ids.size() + c2_ids.size();
    auto status = client_->Flush(collections, progress_monitor);

    // first round set c1 to flushed, c2 to unflushed, progress is 2/5
    // second round set c2 to flushed, progress is 5/5
    std::vector<milvus::Progress> progresses_expected{{(uint32_t)c1_ids.size(), segment_count},
                                                      {segment_count, segment_count}};

    EXPECT_THAT(progresses, ElementsAreArray(progresses_expected));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, FlushGetStateFailure) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::vector<std::string> collections{"c1", "c2"};
    ::milvus::ProgressMonitor progress_monitor{1};
    progress_monitor.SetCheckInterval(1);

    const std::vector<int64_t> c1_ids{1, 2};
    const std::vector<int64_t> c2_ids{101, 102, 103};

    EXPECT_CALL(service_, Flush(_, AllOf(Property(&FlushRequest::collection_names_size, collections.size())), _))
        .WillOnce([&](::grpc::ServerContext*, const FlushRequest*, FlushResponse* response) {
            LongArray c1_segids, c2_segids;
            for (auto id : c1_ids) {
                c1_segids.add_data(id);
            }
            for (auto id : c2_ids) {
                c2_segids.add_data(id);
            }

            auto segids = response->mutable_coll_segids();
            (*segids)[collections[0]] = c1_segids;
            (*segids)[collections[1]] = c2_segids;
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, GetFlushState(_, _, _))
        .Times(1)
        .WillRepeatedly(
            [](::grpc::ServerContext*, const GetFlushStateRequest* request, GetFlushStateResponse* response) {
                response->mutable_status()->set_error_code(::milvus::proto::common::ErrorCode::UnexpectedError);
                return ::grpc::Status{};
            });

    auto status = client_->Flush(collections, progress_monitor);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}
