// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"

using ::milvus::StatusCode;
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

void
FillMinimalV2SearchResults(::milvus::proto::milvus::SearchResults* response) {
    response->mutable_status()->set_code(milvus::proto::common::ErrorCode::Success);
    response->set_session_ts(123456);
    auto* results = response->mutable_results();
    results->set_num_queries(1);
    results->set_top_k(1);
    results->set_primary_field_name("id");
    results->mutable_topks()->Add(1);
    results->mutable_scores()->Add(0.1f);
    results->mutable_ids()->mutable_int_id()->add_data(10000);
}

milvus::SearchRequest
CreateV2SearchRequest() {
    milvus::SearchRequest request;
    request.WithCollectionName("foo");
    request.WithAnnsField("anns_dummy");
    request.WithFilter("id > 0");
    request.WithLimit(1);
    request.AddFloatVector(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f});
    return request;
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, SearchResponseExtraInfoMetadata) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ::milvus::proto::milvus::SearchRequest*,
                     ::milvus::proto::milvus::SearchResults* response) {
            FillMinimalV2SearchResults(response);
            auto* extra_info = response->mutable_status()->mutable_extra_info();
            (*extra_info)["report_value"] = "101";
            (*extra_info)["scanned_remote_bytes"] = "102";
            (*extra_info)["scanned_total_bytes"] = "103";
            (*extra_info)["cache_hit_ratio"] = "0.5";
            return ::grpc::Status{};
        });

    auto request = CreateV2SearchRequest();
    milvus::SearchResponse response;
    auto status = client->Search(request, response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.SessionTs(), 123456u);
    EXPECT_EQ(response.Cost(), 101);
    EXPECT_EQ(response.ScannedRemoteBytes(), 102);
    EXPECT_EQ(response.ScannedTotalBytes(), 103);
    EXPECT_FLOAT_EQ(response.CacheHitRatio(), 0.5f);
    EXPECT_EQ(response.Results().Results().size(), 1);
}

TEST_F(UnconnectMilvusMockedTest, SearchResponseExtraInfoMalformedFallback) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ::milvus::proto::milvus::SearchRequest*,
                     ::milvus::proto::milvus::SearchResults* response) {
            FillMinimalV2SearchResults(response);
            auto* extra_info = response->mutable_status()->mutable_extra_info();
            (*extra_info)["report_value"] = "abc";
            (*extra_info)["scanned_remote_bytes"] = "102";
            (*extra_info)["scanned_total_bytes"] = "bad-total";
            (*extra_info)["cache_hit_ratio"] = "bad-ratio";
            return ::grpc::Status{};
        });

    auto request = CreateV2SearchRequest();
    milvus::SearchResponse response;
    auto status = client->Search(request, response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.SessionTs(), 123456u);
    EXPECT_EQ(response.Cost(), -1);
    EXPECT_EQ(response.ScannedRemoteBytes(), 102);
    EXPECT_EQ(response.ScannedTotalBytes(), -1);
    EXPECT_FLOAT_EQ(response.CacheHitRatio(), -1.0f);
    EXPECT_EQ(response.Results().Results().size(), 1);
}
