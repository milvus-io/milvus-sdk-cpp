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

#include "milvus/MilvusClientV2.h"

class SubSearchRequestTest : public ::testing::Test {};

TEST_F(SubSearchRequestTest, WithMetricType) {
    milvus::SubSearchRequest req;
    auto& ref = req.WithMetricType(milvus::MetricType::L2);
    EXPECT_EQ(req.MetricType(), milvus::MetricType::L2);
    EXPECT_EQ(&ref, &req);
}

TEST_F(SubSearchRequestTest, WithLimit) {
    milvus::SubSearchRequest req;
    auto& ref = req.WithLimit(100);
    EXPECT_EQ(&ref, &req);
}

TEST_F(SubSearchRequestTest, WithFilter) {
    milvus::SubSearchRequest req;
    auto& ref = req.WithFilter("id > 10");
    EXPECT_EQ(req.Filter(), "id > 10");
    EXPECT_EQ(&ref, &req);
}

TEST_F(SubSearchRequestTest, WithAnnsField) {
    milvus::SubSearchRequest req;
    auto& ref = req.WithAnnsField("embedding");
    EXPECT_EQ(req.AnnsField(), "embedding");
    EXPECT_EQ(&ref, &req);
}

TEST_F(SubSearchRequestTest, WithTimezone) {
    milvus::SubSearchRequest req;
    auto& ref = req.WithTimezone("UTC+8");
    EXPECT_EQ(&ref, &req);
}

TEST_F(SubSearchRequestTest, AddFloatVector) {
    milvus::SubSearchRequest req;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    auto& ref = req.AddFloatVector(vec);
    EXPECT_EQ(&ref, &req);
    EXPECT_NE(req.TargetVectors(), nullptr);
}

TEST_F(SubSearchRequestTest, AddBinaryVector) {
    milvus::SubSearchRequest req;
    std::vector<uint8_t> bin_vec = {0xFF, 0x00};
    auto& ref = req.AddBinaryVector(bin_vec);
    EXPECT_EQ(&ref, &req);
}

TEST_F(SubSearchRequestTest, AddSparseVector) {
    milvus::SubSearchRequest req;
    nlohmann::json sparse = {{"1", 0.5}, {"3", 0.8}};
    auto& ref = req.AddSparseVector(sparse);
    EXPECT_EQ(&ref, &req);
}

TEST_F(SubSearchRequestTest, ChainingMethods) {
    milvus::SubSearchRequest req;
    std::vector<float> vec = {1.0f, 2.0f};
    req.WithMetricType(milvus::MetricType::COSINE)
        .WithLimit(50)
        .WithFilter("age > 18")
        .WithAnnsField("vec_field")
        .AddFloatVector(vec);

    EXPECT_EQ(req.MetricType(), milvus::MetricType::COSINE);
    EXPECT_EQ(req.Filter(), "age > 18");
    EXPECT_EQ(req.AnnsField(), "vec_field");
}

TEST_F(SubSearchRequestTest, AddFloat16Vector) {
    milvus::SubSearchRequest req;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    auto& ref = req.AddFloat16Vector(vec);
    EXPECT_EQ(&ref, &req);
    EXPECT_NE(req.TargetVectors(), nullptr);
}

TEST_F(SubSearchRequestTest, AddBFloat16Vector) {
    milvus::SubSearchRequest req;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    auto& ref = req.AddBFloat16Vector(vec);
    EXPECT_EQ(&ref, &req);
    EXPECT_NE(req.TargetVectors(), nullptr);
}

TEST_F(SubSearchRequestTest, AddInt8Vector) {
    milvus::SubSearchRequest req;
    std::vector<int8_t> vec = {1, -2, 3, -4};
    auto& ref = req.AddInt8Vector(vec);
    EXPECT_EQ(&ref, &req);
    EXPECT_NE(req.TargetVectors(), nullptr);
}
