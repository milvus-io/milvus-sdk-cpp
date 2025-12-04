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

#include <cmath>

#include "milvus/utils/FP16.h"

class FP16Test : public ::testing::Test {};

TEST_F(FP16Test, F32toF16) {
    // 0x3C00 represents 1.0 in float16
    EXPECT_EQ(milvus::F32toF16(1.0), 0x3C00);

    // NaN
    EXPECT_EQ(milvus::F32toF16(std::numeric_limits<float>::quiet_NaN()), 0x7E00);

    // Infinity if overflow
    EXPECT_EQ(milvus::F32toF16(std::numeric_limits<float>::infinity()), 0x7C00);
    EXPECT_EQ(milvus::F32toF16(-std::numeric_limits<float>::infinity()), 0xFC00);

    // Largest number
    EXPECT_EQ(milvus::F32toF16(65504), 0x7BFF);
    EXPECT_EQ(milvus::F32toF16(-65504), 0xFBFF);
    EXPECT_EQ(milvus::F32toF16(65505), 0x7BFF);
    EXPECT_EQ(milvus::F32toF16(-65505), 0xFBFF);

    // Smallest number
    EXPECT_EQ(milvus::F32toF16(6.10352e-05), 0x0400);
    EXPECT_EQ(milvus::F32toF16(-6.10352e-05), 0x8400);

    // Zero if underflow
    EXPECT_EQ(milvus::F32toF16(6.10352e-06), 0x0000);
}

TEST_F(FP16Test, F16toF32) {
    // 0x3C00 represents 1.0 in float16
    EXPECT_EQ(milvus::F16toF32(0x3C00), 1.0);

    // NaN
    EXPECT_TRUE(std::isnan(milvus::F16toF32(0x7E00)));

    // Infinity if overflow
    EXPECT_TRUE(std::isinf(milvus::F16toF32(0x7C00)));
    EXPECT_TRUE(std::isinf(milvus::F16toF32(0xFC00)));

    // Largest number
    EXPECT_EQ(milvus::F16toF32(0x7BFF), 65504);
    EXPECT_EQ(milvus::F16toF32(0xFBFF), -65504);

    // Smallest number
    EXPECT_TRUE(std::abs(milvus::F16toF32(0x0400) - 6.10352e-05) < 1e-9);
    EXPECT_TRUE(std::abs(milvus::F16toF32(0x8400) + 6.10352e-05) < 1e-9);

    // Zero if underflow
    EXPECT_TRUE(std::abs(milvus::F16toF32(0x0000)) < 1e-9);
}

TEST_F(FP16Test, F32toBF16) {
    // 0x3F80 represents 1.0 in bfloat16
    EXPECT_EQ(milvus::F32toBF16(1.0), 0x3F80);

    // NaN
    EXPECT_EQ(milvus::F32toBF16(std::numeric_limits<float>::quiet_NaN()), 0x7FC0);

    // Infinity
    EXPECT_EQ(milvus::F32toBF16(std::numeric_limits<float>::infinity()), 0x7F80);
    EXPECT_EQ(milvus::F32toBF16(-std::numeric_limits<float>::infinity()), 0xFF80);
}

TEST_F(FP16Test, BF16toF32) {
    // 0x3F80 represents 1.0 in bfloat16
    EXPECT_TRUE(std::abs(milvus::BF16toF32(0x3F80) - 1.0) < 1e-4);

    // NaN
    EXPECT_TRUE(std::isnan(milvus::BF16toF32(0x7FC0)));

    // Infinity
    EXPECT_TRUE(std::isinf(milvus::BF16toF32(0x7F80)));
    EXPECT_TRUE(std::isinf(milvus::BF16toF32(0xFF80)));
}