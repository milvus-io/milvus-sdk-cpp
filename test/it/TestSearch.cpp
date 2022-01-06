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

#include "TypeUtils.h"
#include "mocks/MilvusMockedTest.h"

using ::milvus::FieldDataPtr;
using ::milvus::StatusCode;
using ::milvus::proto::milvus::SearchRequest;
using ::milvus::proto::milvus::SearchResults;
using ::milvus::proto::schema::DataType;

using ::testing::_;
using ::testing::AllOf;
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

TEST_F(MilvusMockedTest, SearchFoo) {
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
        search_arguments.AddTargetVector(floats);
    }
    search_arguments.SetTravelTimestamp(10000);
    search_arguments.SetGuaranteeTimestamp(10001);
    search_arguments.SetTopK(10);
    search_arguments.SetRoundDecimal(-1);

    EXPECT_CALL(
        service_,
        Search(
            _,
            AllOf(Property(&SearchRequest::collection_name, "foo"), Property(&SearchRequest::dsl, "dummy expression"),
                  Property(&SearchRequest::dsl_type, milvus::proto::common::DslType::BoolExprV1),
                  Property(&SearchRequest::travel_timestamp, 10000),
                  Property(&SearchRequest::guarantee_timestamp, 10001),
                  Property(&SearchRequest::partition_names, UnorderedElementsAre("part1", "part2")),
                  Property(&SearchRequest::output_fields, UnorderedElementsAre("f1", "f2")),
                  Property(&SearchRequest::search_params,
                           UnorderedElementsAre(TestKv("anns_field", "dummy"), TestKv("topk", "10"),
                                                TestKv("metric_type", "dummy"), TestKv("round_decimal", "-1")))),
            _))
        .WillOnce([&floats_vec](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            // check placeholder
            auto placeholder_group_payload = request->placeholder_group();
            milvus::proto::milvus::PlaceholderGroup placeholder_group;
            placeholder_group.ParseFromString(placeholder_group_payload);
            EXPECT_EQ(placeholder_group.placeholders_size(), 1);
            const auto& placeholders = placeholder_group.placeholders(0);
            EXPECT_EQ(placeholders.values_size(), floats_vec.size());
            for (int i = 0; i < floats_vec.size(); ++i) {
                const auto& floats = floats_vec.at(i);
                const auto& placeholder = placeholders.values(i);
                std::vector<float> test_floats(floats.size());
                std::copy_n(placeholder.data(), placeholder.size(), reinterpret_cast<char*>(test_floats.data()));
                EXPECT_EQ(test_floats, floats);
            }

            // TODO(jibin): return dummy data
            response->mutable_status()->set_error_code(milvus::proto::common::ErrorCode::Success);
            return ::grpc::Status{};
        });

    auto status = client_->Search(search_arguments, search_results);
    EXPECT_TRUE(status.IsOk());
    // TODO(jibin): check return data
}

TEST_F(MilvusMockedTest, SearchWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    milvus::SearchArguments search_arguments{};
    milvus::SearchResults search_results{};

    auto status = client_->Search(search_arguments, search_results);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}
