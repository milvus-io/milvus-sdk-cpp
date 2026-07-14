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
#include <string>
#include <vector>

#include "milvus/types/SearchAggregation.h"

namespace {

void
ExpectInvalidArgument(const milvus::Status& status, const std::string& message) {
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), message);
}

}  // namespace

class SearchAggregationTest : public ::testing::Test {};

TEST_F(SearchAggregationTest, RejectsEmptyFields) {
    milvus::SearchAggregation aggregation({}, 1);

    ExpectInvalidArgument(aggregation.Validate(), "SearchAggregation fields cannot be empty");
}

TEST_F(SearchAggregationTest, RejectsEmptyFieldName) {
    milvus::SearchAggregation aggregation({"category", ""}, 1);

    ExpectInvalidArgument(aggregation.Validate(), "SearchAggregation field cannot be empty");
}

TEST_F(SearchAggregationTest, RejectsNonPositiveSize) {
    milvus::SearchAggregation zero_size({"category"}, 0);
    ExpectInvalidArgument(zero_size.Validate(), "SearchAggregation size must be positive");

    milvus::SearchAggregation negative_size({"category"}, -1);
    ExpectInvalidArgument(negative_size.Validate(), "SearchAggregation size must be positive");
}

TEST_F(SearchAggregationTest, RejectsEmptyMetricAlias) {
    milvus::SearchAggregation aggregation({"category"}, 1);
    aggregation.AddMetric("", {milvus::AggregationMetricOp::COUNT, "*"});

    ExpectInvalidArgument(aggregation.Validate(), "SearchAggregation metric alias cannot be empty");
}

TEST_F(SearchAggregationTest, RejectsEmptyMetricFieldName) {
    milvus::SearchAggregation aggregation({"category"}, 1);
    aggregation.AddMetric("count", {milvus::AggregationMetricOp::COUNT, ""});

    ExpectInvalidArgument(aggregation.Validate(), "SearchAggregation metric field name cannot be empty");
}

TEST_F(SearchAggregationTest, RejectsWildcardForNonCountMetrics) {
    const std::vector<milvus::AggregationMetricOp> invalid_ops{
        milvus::AggregationMetricOp::AVG, milvus::AggregationMetricOp::SUM, milvus::AggregationMetricOp::MIN,
        milvus::AggregationMetricOp::MAX};
    for (const auto op : invalid_ops) {
        milvus::SearchAggregation aggregation({"category"}, 1);
        aggregation.AddMetric("metric", {op, "*"});

        ExpectInvalidArgument(aggregation.Validate(), "Only count aggregation supports the '*' field");
    }
}

TEST_F(SearchAggregationTest, RejectsEmptyOrderKey) {
    milvus::SearchAggregation aggregation({"category"}, 1);
    aggregation.AddOrder({"", milvus::AggregationDirection::ASC});

    ExpectInvalidArgument(aggregation.Validate(), "SearchAggregation order key cannot be empty");
}

TEST_F(SearchAggregationTest, RejectsUnknownOrderKey) {
    milvus::SearchAggregation aggregation({"category"}, 1);
    aggregation.AddOrder({"missing_metric", milvus::AggregationDirection::DESC});

    ExpectInvalidArgument(aggregation.Validate(),
                          "SearchAggregation order key must be a metric alias or one of _count/_key");
}

TEST_F(SearchAggregationTest, RejectsInvalidTopHitsSize) {
    auto top_hits = std::make_shared<milvus::AggregationTopHits>(0);
    milvus::SearchAggregation aggregation({"category"}, 1);
    aggregation.WithTopHits(top_hits);

    ExpectInvalidArgument(aggregation.Validate(), "AggregationTopHits size must be positive");
}

TEST_F(SearchAggregationTest, RejectsEmptyTopHitsSortFieldName) {
    auto top_hits = std::make_shared<milvus::AggregationTopHits>(1);
    top_hits->AddSort({"", milvus::AggregationDirection::ASC});
    milvus::SearchAggregation aggregation({"category"}, 1);
    aggregation.WithTopHits(top_hits);

    ExpectInvalidArgument(aggregation.Validate(), "AggregationTopHits sort field name cannot be empty");
}

TEST_F(SearchAggregationTest, RejectsInvalidSubAggregation) {
    auto sub_aggregation = std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{}, 1);
    milvus::SearchAggregation aggregation({"category"}, 1);
    aggregation.WithSubAggregation(sub_aggregation);

    ExpectInvalidArgument(aggregation.Validate(), "SearchAggregation fields cannot be empty");
}

TEST_F(SearchAggregationTest, RejectsCycles) {
    auto aggregation = std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{"category"}, 1);
    aggregation->WithSubAggregation(aggregation);

    ExpectInvalidArgument(aggregation->Validate(), "SearchAggregation cannot contain a cycle");

    aggregation->SetSubAggregation(nullptr);
}

TEST_F(SearchAggregationTest, AcceptsValidConfiguration) {
    auto top_hits = std::make_shared<milvus::AggregationTopHits>(2);
    top_hits->AddSort({"score", milvus::AggregationDirection::DESC});
    auto sub_aggregation = std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{"brand"}, 3);

    milvus::SearchAggregation aggregation({"category"}, 5);
    aggregation.AddMetric("doc_count", {milvus::AggregationMetricOp::COUNT, "*"})
        .AddMetric("avg_price", {milvus::AggregationMetricOp::AVG, "price"})
        .AddOrder({"doc_count", milvus::AggregationDirection::DESC})
        .AddOrder({"_count", milvus::AggregationDirection::DESC})
        .AddOrder({"_key", milvus::AggregationDirection::ASC})
        .WithTopHits(top_hits)
        .WithSubAggregation(sub_aggregation);

    EXPECT_TRUE(aggregation.Validate().IsOk());
}
