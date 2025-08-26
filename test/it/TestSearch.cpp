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

#include "mocks/MilvusMockedTest.h"
#include "utils/CompareUtils.h"
#include "utils/Constants.h"
#include "utils/TypeUtils.h"

using ::milvus::FieldDataPtr;
using ::milvus::StatusCode;
using ::milvus::TestKv;
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

template <typename T>
milvus::Status
DoSearchVectors(testing::StrictMock<milvus::MilvusMockedService>& service_, milvus::MilvusClientPtr& client_,
                std::vector<T> vectors, milvus::SearchResults& search_results, uint64_t simulate_timeout = 0,
                uint64_t search_timeout = 0) {
    milvus::SearchArguments search_arguments{};
    search_arguments.SetCollectionName("foo");
    search_arguments.AddPartitionName("part1");
    search_arguments.AddPartitionName("part2");
    search_arguments.AddOutputField("f1");
    search_arguments.AddOutputField("f2");
    search_arguments.SetFilter("dummy expression");
    for (const auto& vec : vectors) {
        search_arguments.AddTargetVector("anns_dummy", vec);
    }
    search_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    search_arguments.SetLimit(10);
    search_arguments.SetRoundDecimal(1);
    search_arguments.SetMetricType(milvus::MetricType::IP);
    search_arguments.AddExtraParam("nprobe", "10");

    EXPECT_CALL(
        service_,
        Search(
            _,
            AllOf(Property(&SearchRequest::collection_name, "foo"), Property(&SearchRequest::dsl, "dummy expression"),
                  Property(&SearchRequest::dsl_type, milvus::proto::common::DslType::BoolExprV1),
                  Property(&SearchRequest::consistency_level, milvus::proto::common::ConsistencyLevel::Strong),
                  Property(&SearchRequest::guarantee_timestamp, milvus::GuaranteeStrongTs()),
                  Property(&SearchRequest::partition_names, UnorderedElementsAre("part1", "part2")),
                  Property(&SearchRequest::output_fields, UnorderedElementsAre("f1", "f2")),
                  Property(&SearchRequest::search_params,
                           UnorderedElementsAre(TestKv(milvus::ANNS_FIELD, "anns_dummy"), TestKv(milvus::TOPK, "10"),
                                                TestKv(milvus::METRIC_TYPE, "IP"), TestKv(milvus::ROUND_DECIMAL, "1"),
                                                TestKv(milvus::IGNORE_GROWING, "false"), TestKv(milvus::NPROBE, "10"),
                                                TestKv(milvus::OFFSET, "0"), _))),
            _))
        .WillOnce([&vectors, simulate_timeout](::grpc::ServerContext*, const SearchRequest* request,
                                               SearchResults* response) {
            // check placeholder
            const auto& placeholder_group_payload = request->placeholder_group();
            milvus::proto::common::PlaceholderGroup placeholder_group;
            placeholder_group.ParseFromString(placeholder_group_payload);
            EXPECT_EQ(placeholder_group.placeholders_size(), 1);
            const auto& placeholders = placeholder_group.placeholders(0);
            EXPECT_EQ(placeholders.values_size(), vectors.size());
            for (int i = 0; i < vectors.size(); ++i) {
                const auto& vector = vectors.at(i);
                const auto& placeholder = placeholders.values(i);
                T test_vector(vector.size());
                std::copy_n(placeholder.data(), placeholder.size(), reinterpret_cast<char*>(test_vector.data()));
                EXPECT_EQ(test_vector, vector);
            }

            response->mutable_status()->set_code(milvus::proto::common::ErrorCode::Success);
            auto* results = response->mutable_results();
            results->set_top_k(10);
            results->set_num_queries(2);
            results->set_primary_field_name("PrimaryKey");

            auto* fields_f1 = results->add_fields_data();
            fields_f1->set_field_id(1000);
            fields_f1->set_field_name("f1");
            fields_f1->set_type(milvus::proto::schema::DataType::Bool);
            std::vector<bool> out_f1{true, false, false, true, false};
            for (const auto val : out_f1) {
                fields_f1->mutable_scalars()->mutable_bool_data()->add_data(val);
            }

            auto* fields_f2 = results->add_fields_data();
            fields_f2->set_field_id(1001);
            fields_f2->set_field_name("f2");
            fields_f2->set_type(milvus::proto::schema::DataType::Int16);
            std::vector<int16_t> out_f2{1, 2, 3, 4, 5};
            for (const auto val : out_f2) {
                fields_f2->mutable_scalars()->mutable_int_data()->add_data(val);
            }

            // ids, topk and scores
            results->mutable_topks()->Add(2);
            results->mutable_topks()->Add(3);
            results->mutable_scores()->Add(0.1f);
            results->mutable_scores()->Add(0.2f);
            results->mutable_scores()->Add(0.3f);
            results->mutable_scores()->Add(0.4f);
            results->mutable_scores()->Add(0.5f);
            results->mutable_ids()->mutable_int_id()->add_data(10000);
            results->mutable_ids()->mutable_int_id()->add_data(20000);
            results->mutable_ids()->mutable_int_id()->add_data(30000);
            results->mutable_ids()->mutable_int_id()->add_data(40000);
            results->mutable_ids()->mutable_int_id()->add_data(50000);

            // sleep if timeout
            if (simulate_timeout > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds{simulate_timeout});
            }

            return ::grpc::Status{};
        });

    client_->SetRpcDeadlineMs(search_timeout);
    return client_->Search(search_arguments, search_results);
}

template <typename T>
void
TestSearchVectors(testing::StrictMock<milvus::MilvusMockedService>& service_, milvus::MilvusClientPtr& client_,
                  std::vector<T> vectors) {
    milvus::SearchResults search_results{};

    auto status = DoSearchVectors<T>(service_, client_, vectors, search_results);
    EXPECT_TRUE(status.IsOk());
    auto& results = search_results.Results();
    EXPECT_EQ(results.size(), 2);

    // check result in column-based
    auto single_1 = results.at(0);
    auto ids_1 = std::vector<int64_t>{10000, 20000};
    auto scores_1 = std::vector<float>{0.1f, 0.2f};
    auto f1_1 = std::vector<bool>{true, false};
    auto f2_1 = std::vector<int16_t>{1, 2};
    EXPECT_EQ(single_1.Ids().IntIDArray(), ids_1);
    EXPECT_EQ(single_1.Scores(), scores_1);
    EXPECT_EQ(single_1.OutputFields().size(), 2);
    EXPECT_EQ(std::static_pointer_cast<milvus::BoolFieldData>(single_1.OutputField("f1"))->Data(), f1_1);
    EXPECT_EQ(std::static_pointer_cast<milvus::Int16FieldData>(single_1.OutputField("f2"))->Data(), f2_1);

    auto single_2 = results.at(1);
    auto ids_2 = std::vector<int64_t>{30000, 40000, 50000};
    auto scores_2 = std::vector<float>{0.3f, 0.4f, 0.5f};
    auto f1_2 = std::vector<bool>{false, true, false};
    auto f2_2 = std::vector<int16_t>{3, 4, 5};
    EXPECT_EQ(single_2.Ids().IntIDArray(), ids_2);
    EXPECT_EQ(single_2.Scores(), scores_2);
    EXPECT_EQ(single_2.OutputFields().size(), 2);
    EXPECT_EQ(std::static_pointer_cast<milvus::BoolFieldData>(single_2.OutputField("f1"))->Data(), f1_2);
    EXPECT_EQ(std::static_pointer_cast<milvus::Int16FieldData>(single_2.OutputField("f2"))->Data(), f2_2);

    // check result in row-based
    EXPECT_EQ(single_1.GetRowCount(), 2);
    EXPECT_EQ(single_2.GetRowCount(), 3);
    EXPECT_EQ(single_1.PrimaryKeyName(), "PrimaryKey");
    EXPECT_EQ(single_2.PrimaryKeyName(), "PrimaryKey");

    std::vector<nlohmann::json> rows;
    status = single_1.OutputRows(rows);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rows.size(), 2);
    status = single_2.OutputRows(rows);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rows.size(), 3);

    for (auto i = 0; i < single_1.GetRowCount(); i++) {
        nlohmann::json row;
        status = single_1.OutputRow(i, row);
        EXPECT_TRUE(status.IsOk());
        EXPECT_TRUE(row.contains("PrimaryKey"));
        EXPECT_EQ(row["PrimaryKey"].get<int64_t>(), ids_1.at(i));
        EXPECT_TRUE(row.contains(milvus::SCORE));
        EXPECT_EQ(row[milvus::SCORE].get<float>(), scores_1.at(i));
        EXPECT_TRUE(row.contains("f1"));
        EXPECT_EQ(row["f1"].get<bool>(), f1_1.at(i));
        EXPECT_TRUE(row.contains("f2"));
        EXPECT_EQ(row["f2"].get<int16_t>(), f2_1.at(i));
    }
    for (auto i = 0; i < single_2.GetRowCount(); i++) {
        nlohmann::json row;
        status = single_2.OutputRow(i, row);
        EXPECT_TRUE(status.IsOk());
        EXPECT_TRUE(row.contains("PrimaryKey"));
        EXPECT_EQ(row["PrimaryKey"].get<int64_t>(), ids_2.at(i));
        EXPECT_TRUE(row.contains(milvus::SCORE));
        EXPECT_EQ(row[milvus::SCORE].get<float>(), scores_2.at(i));
        EXPECT_TRUE(row.contains("f1"));
        EXPECT_EQ(row["f1"].get<bool>(), f1_2.at(i));
        EXPECT_TRUE(row.contains("f2"));
        EXPECT_EQ(row["f2"].get<int16_t>(), f2_2.at(i));
    }
}

TEST_F(MilvusMockedTest, Search) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::vector<std::vector<float>> float_vectors = {std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
                                                     std::vector<float>{0.2f, 0.3f, 0.4f, 0.5f}};
    TestSearchVectors<std::vector<float>>(service_, client_, float_vectors);

    std::vector<std::vector<uint8_t>> bin_vectors = {std::vector<uint8_t>{1, 2, 3, 4},
                                                     std::vector<uint8_t>{2, 3, 4, 5}};
    TestSearchVectors<std::vector<uint8_t>>(service_, client_, bin_vectors);
}

TEST_F(MilvusMockedTest, SearchWithTimeoutExpired) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::vector<std::vector<float>> float_vectors = {std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
                                                     std::vector<float>{0.2f, 0.3f, 0.4f, 0.5f}};
    milvus::SearchResults search_results{};

    auto t0 = std::chrono::system_clock::now();
    auto status = DoSearchVectors(service_, client_, float_vectors, search_results, 1000, 500);
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t0).count();

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::TIMEOUT);
    EXPECT_GE(duration, 500);
}

TEST_F(MilvusMockedTest, SearchWithTimeoutOk) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::vector<std::vector<float>> float_vectors = {std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
                                                     std::vector<float>{0.2f, 0.3f, 0.4f, 0.5f}};
    milvus::SearchResults search_results{};

    auto t0 = std::chrono::system_clock::now();
    auto status = DoSearchVectors(service_, client_, float_vectors, search_results, 500, 1000);
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t0).count();

    EXPECT_TRUE(status.IsOk());
    EXPECT_GE(duration, 500);
}

TEST_F(UnconnectMilvusMockedTest, SearchWithoutConnect) {
    milvus::SearchArguments search_arguments{};
    milvus::SearchResults search_results{};

    auto status = client_->Search(search_arguments, search_results);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}
