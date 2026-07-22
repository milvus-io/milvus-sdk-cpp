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

TEST_F(UnconnectMilvusMockedTest, SearchAggregation) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ::milvus::proto::milvus::SearchRequest* request,
                     ::milvus::proto::milvus::SearchResults* response) {
            EXPECT_TRUE(request->has_search_aggregation());
            const auto& aggregation = request->search_aggregation();
            EXPECT_EQ(aggregation.fields_size(), 2);
            EXPECT_EQ(aggregation.fields(0), "category");
            EXPECT_EQ(aggregation.size(), 5);
            EXPECT_EQ(aggregation.metrics().at("doc_count").op(), "count");
            EXPECT_EQ(aggregation.order(0).key(), "doc_count");
            EXPECT_EQ(aggregation.top_hits().sort(0).field_name(), "_score");
            EXPECT_EQ(aggregation.sub_aggregation().fields(0), "brand");

            FillMinimalV2SearchResults(response);
            auto bucket = response->mutable_results()->add_agg_buckets();
            bucket->set_count(7);
            auto key = bucket->add_key();
            key->set_field_id(101);
            key->set_field_name("category");
            key->set_string_val("books");
            (*bucket->mutable_metrics())["doc_count"].set_int_val(7);
            auto hit = bucket->add_hits();
            hit->set_int_pk(10000);
            hit->set_score(0.91f);
            auto title = hit->add_fields();
            title->set_field_id(201);
            title->set_field_name("title");
            title->set_string_val("Book A");
            auto payload = hit->add_fields();
            payload->set_field_id(202);
            payload->set_field_name("payload");
            payload->set_bytes_val(std::string("\x01\x02\x03", 3));
            auto sub_group = bucket->add_sub_groups();
            sub_group->set_count(3);
            auto sub_key = sub_group->add_key();
            sub_key->set_field_name("brand");
            sub_key->set_string_val("acme");
            response->mutable_results()->add_agg_topks(1);
            return ::grpc::Status{};
        });

    auto top_hits = std::make_shared<milvus::AggregationTopHits>(2);
    top_hits->AddSort({"_score", milvus::AggregationDirection::DESC});
    auto sub_aggregation = std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{"brand"}, 3);
    auto aggregation = std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{"category", "region"}, 5);
    aggregation->AddMetric("doc_count", {milvus::AggregationMetricOp::COUNT, "*"})
        .AddOrder({"doc_count", milvus::AggregationDirection::DESC})
        .WithTopHits(top_hits)
        .WithSubAggregation(sub_aggregation);

    auto request = CreateV2SearchRequest();
    request.WithSearchAggregation(aggregation);
    milvus::SearchResponse response;
    auto status = client->Search(request, response);

    ASSERT_TRUE(status.IsOk());
    ASSERT_EQ(response.AggregationBuckets().size(), 1);
    const auto& bucket = response.AggregationBuckets().at(0).at(0);
    EXPECT_EQ(bucket.count, 7);
    EXPECT_EQ(bucket.key.at(0).value.get<std::string>(), "books");
    EXPECT_EQ(bucket.metrics.at("doc_count").get<int64_t>(), 7);
    EXPECT_EQ(bucket.hits.at(0).id.get<int64_t>(), 10000);
    EXPECT_EQ(bucket.hits.at(0).fields.at("title").get<std::string>(), "Book A");
    EXPECT_TRUE(bucket.hits.at(0).fields.at("payload").is_binary());
    EXPECT_EQ(bucket.sub_groups.at(0).key.at(0).value.get<std::string>(), "acme");
}

TEST_F(UnconnectMilvusMockedTest, SearchAggregationConversionFailureLeavesResponseUnchanged) {
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
            response->mutable_results()->add_agg_buckets()->set_count(7);
            response->mutable_results()->add_agg_topks(1);
            return ::grpc::Status{};
        })
        .WillOnce([](::grpc::ServerContext*, const ::milvus::proto::milvus::SearchRequest*,
                     ::milvus::proto::milvus::SearchResults* response) {
            FillMinimalV2SearchResults(response);
            response->set_session_ts(654321);
            response->mutable_results()->mutable_ids()->mutable_int_id()->set_data(0, 20000);
            auto* extra_info = response->mutable_status()->mutable_extra_info();
            (*extra_info)["report_value"] = "201";
            (*extra_info)["scanned_remote_bytes"] = "202";
            (*extra_info)["scanned_total_bytes"] = "203";
            (*extra_info)["cache_hit_ratio"] = "0.75";
            response->mutable_results()->add_agg_buckets()->set_count(8);
            response->mutable_results()->add_agg_topks(2);
            return ::grpc::Status{};
        });

    auto request = CreateV2SearchRequest();
    milvus::SearchResponse response;
    ASSERT_TRUE(client->Search(request, response).IsOk());
    ASSERT_EQ(response.Results().Results().size(), 1);
    ASSERT_EQ(response.AggregationBuckets().size(), 1);
    ASSERT_EQ(response.AggregationBuckets().at(0).size(), 1);

    auto status = client->Search(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(response.Results().Results().at(0).Ids().IntIDArray(), (std::vector<int64_t>{10000}));
    EXPECT_EQ(response.AggregationBuckets().at(0).at(0).count, 7);
    EXPECT_EQ(response.SessionTs(), 123456u);
    EXPECT_EQ(response.Cost(), 101);
    EXPECT_EQ(response.ScannedRemoteBytes(), 102);
    EXPECT_EQ(response.ScannedTotalBytes(), 103);
    EXPECT_FLOAT_EQ(response.CacheHitRatio(), 0.5f);
}
