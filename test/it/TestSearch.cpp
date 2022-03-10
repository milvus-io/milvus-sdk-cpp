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
#include <nlohmann/json.hpp>

#include "TypeUtils.h"
#include "mocks/MilvusMockedTest.h"

using ::milvus::FieldDataPtr;
using ::milvus::StatusCode;
using ::milvus::proto::milvus::DescribeCollectionRequest;
using ::milvus::proto::milvus::DescribeCollectionResponse;
using ::milvus::proto::milvus::SearchRequest;
using ::milvus::proto::milvus::SearchResults;
using ::milvus::proto::schema::DataType;

using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

namespace {
struct TestKv {
    TestKv(const std::string& key, const std::string& value) : key_(key), value_(value) {
    }
    std::string key_;
    std::string value_;
};

bool
operator==(const milvus::proto::common::KeyValuePair& lhs, const TestKv& rhs) {
    return lhs.key() == rhs.key_ && lhs.value() == rhs.value_;
}
}  // namespace

template <typename T>
void
TestSearchVectors(testing::StrictMock<::milvus::MilvusMockedService>& service_,
                  std::shared_ptr<::milvus::MilvusClient>& client_, std::vector<std::vector<T>> vectors) {
    milvus::SearchArguments search_arguments{};
    milvus::SearchResults search_results{};
    search_arguments.SetCollectionName("foo");
    search_arguments.AddPartitionName("part1");
    search_arguments.AddPartitionName("part2");
    search_arguments.AddOutputField("f1");
    search_arguments.AddOutputField("f2");
    search_arguments.SetExpression("dummy expression");
    for (const auto& vec : vectors) {
        search_arguments.AddTargetVector("anns_dummy", vec);
    }
    search_arguments.SetTravelTimestamp(10000);
    search_arguments.SetGuaranteeTimestamp(milvus::GuaranteeStrongTs());
    search_arguments.SetTopK(10);
    search_arguments.SetRoundDecimal(-1);
    search_arguments.SetMetricType(milvus::MetricType::IP);

    search_arguments.AddExtraParam("nprobe", 10);

    EXPECT_CALL(service_, DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, "foo"), _))
        .WillOnce([](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
            response->set_collectionid(1000);
            response->set_shards_num(2);
            response->set_created_timestamp(10000);
            auto* proto_schema = response->mutable_schema();
            auto* field = proto_schema->add_fields();
            field->set_data_type(DataType::FloatVector);
            field->set_name("anns_dummy");
            return ::grpc::Status{};
        });

    EXPECT_CALL(
        service_,
        Search(
            _,
            AllOf(Property(&SearchRequest::collection_name, "foo"), Property(&SearchRequest::dsl, "dummy expression"),
                  Property(&SearchRequest::dsl_type, milvus::proto::common::DslType::BoolExprV1),
                  Property(&SearchRequest::travel_timestamp, 10000),
                  Property(&SearchRequest::guarantee_timestamp, milvus::GuaranteeStrongTs()),
                  Property(&SearchRequest::partition_names, UnorderedElementsAre("part1", "part2")),
                  Property(&SearchRequest::output_fields, UnorderedElementsAre("f1", "f2")),
                  Property(&SearchRequest::search_params,
                           UnorderedElementsAre(TestKv("anns_field", "anns_dummy"), TestKv("topk", "10"),
                                                TestKv("metric_type", "IP"), TestKv("round_decimal", "-1"), _))),
            _))
        .WillOnce([&vectors](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            // check params
            auto& search_params = request->search_params();
            std::string extra_params_payload;
            for (const auto& pair : search_params) {
                if (pair.key() == "params") {
                    extra_params_payload = pair.value();
                }
            }
            auto json_params = nlohmann::json::parse(extra_params_payload);
            EXPECT_EQ(json_params["nprobe"].get<int>(), 10);

            // check placeholder
            auto placeholder_group_payload = request->placeholder_group();
            milvus::proto::milvus::PlaceholderGroup placeholder_group;
            placeholder_group.ParseFromString(placeholder_group_payload);
            EXPECT_EQ(placeholder_group.placeholders_size(), 1);
            const auto& placeholders = placeholder_group.placeholders(0);
            EXPECT_EQ(placeholders.values_size(), vectors.size());
            for (int i = 0; i < vectors.size(); ++i) {
                const auto& vector = vectors.at(i);
                const auto& placeholder = placeholders.values(i);
                std::vector<T> test_vector(vector.size());
                std::copy_n(placeholder.data(), placeholder.size(), reinterpret_cast<char*>(test_vector.data()));
                EXPECT_EQ(test_vector, vector);
            }

            response->mutable_status()->set_error_code(milvus::proto::common::ErrorCode::Success);
            auto* results = response->mutable_results();
            results->set_top_k(10);
            results->set_num_queries(2);

            auto* fields_f1 = results->add_fields_data();
            fields_f1->set_field_id(1000);
            fields_f1->set_field_name("f1");
            fields_f1->set_type(milvus::proto::schema::DataType::Bool);
            std::vector<bool> out_f1{true, false, false, true};
            for (const auto val : out_f1) {
                fields_f1->mutable_scalars()->mutable_bool_data()->add_data(val);
            }

            auto* fields_f2 = results->add_fields_data();
            fields_f2->set_field_id(1001);
            fields_f2->set_field_name("f2");
            fields_f2->set_type(milvus::proto::schema::DataType::Int16);
            std::vector<int16_t> out_f2{1, 2, 3, 4};
            for (const auto val : out_f2) {
                fields_f2->mutable_scalars()->mutable_int_data()->add_data(val);
            }

            // ids, topk and scores
            results->mutable_topks()->Add(2);
            results->mutable_topks()->Add(2);
            results->mutable_scores()->Add(0.1f);
            results->mutable_scores()->Add(0.2f);
            results->mutable_scores()->Add(0.3f);
            results->mutable_scores()->Add(0.4f);
            results->mutable_ids()->mutable_int_id()->add_data(10000);
            results->mutable_ids()->mutable_int_id()->add_data(20000);
            results->mutable_ids()->mutable_int_id()->add_data(30000);
            results->mutable_ids()->mutable_int_id()->add_data(40000);

            return ::grpc::Status{};
        });

    auto status = client_->Search(search_arguments, search_results);
    EXPECT_TRUE(status.IsOk());
    auto& results = search_results.Results();
    EXPECT_EQ(results.size(), 2);
    EXPECT_THAT(results.at(0).Ids().IntIDArray(), ElementsAre(10000, 20000));
    EXPECT_THAT(results.at(0).Scores(), ElementsAre(0.1f, 0.2f));
    EXPECT_THAT(std::static_pointer_cast<milvus::BoolFieldData>(results.at(0).OutputField("f1"))->Data(),
                ElementsAre(true, false));
    EXPECT_THAT(std::static_pointer_cast<milvus::Int16FieldData>(results.at(0).OutputField("f2"))->Data(),
                ElementsAre(1, 2));

    EXPECT_THAT(results.at(1).Ids().IntIDArray(), ElementsAre(30000, 40000));
    EXPECT_THAT(results.at(1).Scores(), ElementsAre(0.3f, 0.4f));
    EXPECT_THAT(std::static_pointer_cast<milvus::BoolFieldData>(results.at(1).OutputField("f1"))->Data(),
                ElementsAre(false, true));
    EXPECT_THAT(std::static_pointer_cast<milvus::Int16FieldData>(results.at(1).OutputField("f2"))->Data(),
                ElementsAre(3, 4));
}

TEST_F(MilvusMockedTest, SearchFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::vector<std::vector<float>> float_vectors = {std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
                                                     std::vector<float>{0.2f, 0.3f, 0.4f, 0.5f}};
    TestSearchVectors<float>(service_, client_, float_vectors);

    std::vector<std::vector<uint8_t>> bin_vectors = {std::vector<uint8_t>{1, 2, 3, 4},
                                                     std::vector<uint8_t>{2, 3, 4, 5}};
    TestSearchVectors<uint8_t>(service_, client_, bin_vectors);
}

TEST_F(MilvusMockedTest, SearchWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    milvus::SearchArguments search_arguments{};
    milvus::SearchResults search_results{};

    auto status = client_->Search(search_arguments, search_results);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, SearchWithMismatchedAnnsField) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::vector<std::vector<float>> floats_vec = {std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
                                                  std::vector<float>{0.2f, 0.3f, 0.4f, 0.5f}};

    milvus::SearchArguments search_arguments{};
    milvus::SearchResults search_results{};
    search_arguments.SetCollectionName("foo");
    search_arguments.AddPartitionName("part1");
    search_arguments.AddPartitionName("part2");
    search_arguments.AddOutputField("f1");
    search_arguments.AddOutputField("f2");
    search_arguments.SetExpression("dummy expression");
    for (const auto& floats : floats_vec) {
        search_arguments.AddTargetVector("anns_dummy", floats);
    }
    search_arguments.SetTravelTimestamp(10000);
    search_arguments.SetGuaranteeTimestamp(10001);
    search_arguments.SetTopK(10);
    search_arguments.SetRoundDecimal(-1);

    EXPECT_CALL(service_, DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, "foo"), _))
        .WillOnce([](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
            response->set_collectionid(1000);
            response->set_shards_num(2);
            response->set_created_timestamp(10000);
            auto* proto_schema = response->mutable_schema();
            auto* field = proto_schema->add_fields();
            field->set_data_type(DataType::FloatVector);
            field->set_name("anns_mismatch");
            return ::grpc::Status{};
        });

    auto status = client_->Search(search_arguments, search_results);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_AGUMENT);
}