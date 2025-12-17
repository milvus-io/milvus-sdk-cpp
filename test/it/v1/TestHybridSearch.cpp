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
#include <memory>
#include <nlohmann/json.hpp>
#include <thread>
#include <utility>

#include "../mocks/MilvusMockedTest.h"
#include "utils/CompareUtils.h"
#include "utils/Constants.h"
#include "utils/ExtraParamUtils.h"
#include "utils/TypeUtils.h"

using ::milvus::StatusCode;
using ::milvus::TestKv;
using ::milvus::proto::milvus::HybridSearchRequest;
using ::milvus::proto::milvus::SearchResults;
using ::milvus::proto::schema::DataType;

using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

TEST_F(MilvusMockedTest, HybridSearch) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    milvus::HybridSearchArguments search_arg{};
    search_arg.SetCollectionName("foo");
    search_arg.AddPartitionName("part1");
    search_arg.AddPartitionName("part2");
    search_arg.AddOutputField("f1");
    search_arg.AddOutputField("f2");
    search_arg.SetDatabaseName("db");
    search_arg.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);
    search_arg.SetLimit(3);
    search_arg.SetOffset(5);
    search_arg.SetRoundDecimal(1);
    search_arg.SetIgnoreGrowing(true);

    auto sub_req1 = std::make_shared<milvus::SubSearchRequest>();
    sub_req1->SetLimit(5);
    sub_req1->SetFilter("dummy expression");
    sub_req1->SetMetricType(milvus::MetricType::COSINE);
    sub_req1->SetRadius(0.7);
    sub_req1->SetRangeFilter(1.0);
    sub_req1->SetAnnsField("dense");
    auto dense_vector = std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f};
    sub_req1->AddFloatVector(dense_vector);
    search_arg.AddSubRequest(sub_req1);

    auto sub_req2 = std::make_shared<milvus::SubSearchRequest>();
    sub_req2->SetLimit(15);
    sub_req2->SetFilter("dummy expression");
    sub_req2->SetMetricType(milvus::MetricType::HAMMING);
    sub_req2->SetRadius(3.7);
    sub_req2->SetRangeFilter(2.0);
    sub_req2->SetAnnsField("bin");
    auto bin_vector = std::vector<uint8_t>{1, 2, 3, 4};
    sub_req2->AddBinaryVector(bin_vector);
    search_arg.AddSubRequest(sub_req2);

    auto reranker = std::make_shared<milvus::RRFRerank>(90);
    search_arg.SetRerank(reranker);

    auto expected_ids = std::vector<std::string>{"a", "bb", "ccc"};
    auto expected_scores = std::vector<float>{0.5f, 0.4f, 0.3f};
    auto expected_f1 = std::vector<bool>{true, false, false};
    auto expected_f2 = std::vector<int16_t>{1, 2, 3};
    EXPECT_CALL(
        service_,
        HybridSearch(
            _,
            AllOf(Property(&HybridSearchRequest::collection_name, search_arg.CollectionName()),
                  Property(&HybridSearchRequest::partition_names, UnorderedElementsAre("part1", "part2")),
                  Property(&HybridSearchRequest::db_name, search_arg.DatabaseName()),
                  Property(&HybridSearchRequest::consistency_level, milvus::proto::common::ConsistencyLevel::Bounded),
                  Property(&HybridSearchRequest::output_fields, UnorderedElementsAre("f1", "f2")),
                  Property(
                      &HybridSearchRequest::rank_params,
                      UnorderedElementsAre(TestKv(milvus::LIMIT, "3"), TestKv(milvus::OFFSET, "5"),
                                           TestKv(milvus::ROUND_DECIMAL, "1"), TestKv(milvus::IGNORE_GROWING, "true"),
                                           TestKv(milvus::STRATEGY, "rrf"), TestKv(milvus::PARAMS, "{\"k\":90}")))),
            _))
        .WillOnce([&search_arg, &dense_vector, &bin_vector, &expected_ids, &expected_scores, &expected_f1,
                   &expected_f2](::grpc::ServerContext*, const HybridSearchRequest* request, SearchResults* response) {
            // check more inputs
            const auto& sub_reqs = search_arg.SubRequests();
            const auto& rpc_sub_reqs = request->requests();
            EXPECT_EQ(sub_reqs.size(), rpc_sub_reqs.size());
            for (auto i = 0; i < sub_reqs.size(); i++) {
                auto sub_req = sub_reqs[i];
                auto rpc_sub_req = rpc_sub_reqs.at(i);
                EXPECT_EQ(rpc_sub_req.dsl(), "dummy expression");
                EXPECT_EQ(rpc_sub_req.dsl_type(), ::milvus::proto::common::DslType::BoolExprV1);
                auto rpc_search_params = rpc_sub_req.search_params();
                EXPECT_THAT(rpc_search_params,
                            UnorderedElementsAre(
                                TestKv(milvus::ANNS_FIELD, sub_req->AnnsField()),
                                TestKv(milvus::TOPK, std::to_string(sub_req->Limit())),
                                TestKv(milvus::METRIC_TYPE, std::to_string(sub_req->MetricType())),
                                TestKv(milvus::RADIUS, milvus::DoubleToString(sub_req->Radius())),
                                TestKv(milvus::RANGE_FILTER, milvus::DoubleToString(sub_req->RangeFilter())), _));

                const auto& placeholder_group_payload = rpc_sub_req.placeholder_group();
                milvus::proto::common::PlaceholderGroup placeholder_group;
                placeholder_group.ParseFromString(placeholder_group_payload);
                EXPECT_EQ(placeholder_group.placeholders_size(), 1);
                const auto& placeholders = placeholder_group.placeholders(0);
                EXPECT_EQ(placeholders.values_size(), 1);
                const auto& placeholder = placeholders.values(0);
                if (sub_req->AnnsField() == "dense") {
                    std::vector<float> test_vector(dense_vector.size());
                    std::copy_n(placeholder.data(), placeholder.size(), reinterpret_cast<char*>(test_vector.data()));
                    EXPECT_EQ(test_vector, dense_vector);
                } else {
                    std::vector<uint8_t> test_vector(bin_vector.size());
                    std::copy_n(placeholder.data(), placeholder.size(), reinterpret_cast<char*>(test_vector.data()));
                    EXPECT_EQ(test_vector, bin_vector);
                }
            }

            // set results
            response->mutable_status()->set_code(milvus::proto::common::ErrorCode::Success);
            auto* results = response->mutable_results();
            results->set_top_k(expected_ids.size());
            results->set_num_queries(1);
            results->set_primary_field_name("PrimaryKey");

            auto* fields_f1 = results->add_fields_data();
            fields_f1->set_field_id(1000);
            fields_f1->set_field_name("f1");
            fields_f1->set_type(milvus::proto::schema::DataType::Bool);
            for (const auto val : expected_f1) {
                fields_f1->mutable_scalars()->mutable_bool_data()->add_data(val);
            }

            auto* fields_f2 = results->add_fields_data();
            fields_f2->set_field_id(1001);
            fields_f2->set_field_name("f2");
            fields_f2->set_type(milvus::proto::schema::DataType::Int16);
            for (const auto val : expected_f2) {
                fields_f2->mutable_scalars()->mutable_int_data()->add_data(val);
            }

            // ids, topk and scores
            results->mutable_topks()->Add(3);
            for (auto id : expected_ids) {
                results->mutable_ids()->mutable_str_id()->add_data(id);
            }
            for (auto score : expected_scores) {
                results->mutable_scores()->Add(score);
            }

            return ::grpc::Status{};
        });

    milvus::SearchResults search_results;
    auto status = client_->HybridSearch(search_arg, search_results);

    EXPECT_TRUE(status.IsOk());
    auto& results = search_results.Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results.at(0).Ids().StrIDArray(), expected_ids);
    EXPECT_EQ(results.at(0).Scores(), expected_scores);
    EXPECT_EQ(results.at(0).OutputField<milvus::BoolFieldData>("f1")->Data(), expected_f1);
    EXPECT_EQ(results.at(0).OutputField<milvus::Int16FieldData>("f2")->Data(), expected_f2);
}
