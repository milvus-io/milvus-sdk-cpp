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
#include <limits>

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

TEST_F(FP16Test, F32toF16Roundtrip) {
    // Normal values
    float val = 1.5f;
    uint16_t fp16 = milvus::F32toF16(val);
    float result = milvus::F16toF32(fp16);
    EXPECT_NEAR(result, val, 0.001f);

    val = -3.14f;
    fp16 = milvus::F32toF16(val);
    result = milvus::F16toF32(fp16);
    EXPECT_NEAR(result, val, 0.01f);
}

TEST_F(FP16Test, F32toBF16Roundtrip) {
    float val = 1.5f;
    uint16_t bf16 = milvus::F32toBF16(val);
    float result = milvus::BF16toF32(bf16);
    EXPECT_NEAR(result, val, 0.01f);

    val = -3.14f;
    bf16 = milvus::F32toBF16(val);
    result = milvus::BF16toF32(bf16);
    EXPECT_NEAR(result, val, 0.02f);
}

TEST_F(FP16Test, ArrayF32toF16Roundtrip) {
    std::vector<float> input{0.0f, 1.0f, -1.0f, 0.5f, 100.0f};
    auto fp16_array = milvus::ArrayF32toF16(input);
    EXPECT_EQ(fp16_array.size(), input.size());

    auto output = milvus::ArrayF16toF32(fp16_array);
    EXPECT_EQ(output.size(), input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(output[i], input[i], 0.1f);
    }
}

TEST_F(FP16Test, ArrayF32toBF16Roundtrip) {
    std::vector<float> input{0.0f, 1.0f, -1.0f, 0.5f, 100.0f};
    auto bf16_array = milvus::ArrayF32toBF16(input);
    EXPECT_EQ(bf16_array.size(), input.size());

    auto output = milvus::ArrayBF16toF32(bf16_array);
    EXPECT_EQ(output.size(), input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(output[i], input[i], 1.0f);
    }
}

TEST_F(FP16Test, ZeroValue) {
    // FP16
    uint16_t fp16 = milvus::F32toF16(0.0f);
    float result = milvus::F16toF32(fp16);
    EXPECT_FLOAT_EQ(result, 0.0f);

    // BF16
    uint16_t bf16 = milvus::F32toBF16(0.0f);
    result = milvus::BF16toF32(bf16);
    EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST_F(FP16Test, Infinity) {
    float pos_inf = std::numeric_limits<float>::infinity();
    float neg_inf = -std::numeric_limits<float>::infinity();

    // FP16 infinity
    uint16_t fp16_pos = milvus::F32toF16(pos_inf);
    float result_pos = milvus::F16toF32(fp16_pos);
    EXPECT_TRUE(std::isinf(result_pos) && result_pos > 0);

    uint16_t fp16_neg = milvus::F32toF16(neg_inf);
    float result_neg = milvus::F16toF32(fp16_neg);
    EXPECT_TRUE(std::isinf(result_neg) && result_neg < 0);

    // BF16 infinity
    uint16_t bf16_pos = milvus::F32toBF16(pos_inf);
    result_pos = milvus::BF16toF32(bf16_pos);
    EXPECT_TRUE(std::isinf(result_pos) && result_pos > 0);

    uint16_t bf16_neg = milvus::F32toBF16(neg_inf);
    result_neg = milvus::BF16toF32(bf16_neg);
    EXPECT_TRUE(std::isinf(result_neg) && result_neg < 0);
}

TEST_F(FP16Test, NaN) {
    float nan_val = std::numeric_limits<float>::quiet_NaN();

    // FP16 NaN
    uint16_t fp16 = milvus::F32toF16(nan_val);
    float result = milvus::F16toF32(fp16);
    EXPECT_TRUE(std::isnan(result));

    // BF16 NaN
    uint16_t bf16 = milvus::F32toBF16(nan_val);
    result = milvus::BF16toF32(bf16);
    EXPECT_TRUE(std::isnan(result));
}

TEST_F(FP16Test, VerySmallValue) {
    // FP16 denormalized minimum ~5.96e-8
    float small = 0.0001f;
    uint16_t fp16 = milvus::F32toF16(small);
    float result = milvus::F16toF32(fp16);
    EXPECT_NEAR(result, small, small * 0.1f);

    // BF16
    uint16_t bf16 = milvus::F32toBF16(small);
    result = milvus::BF16toF32(bf16);
    EXPECT_NEAR(result, small, small * 0.2f);
}

TEST_F(FP16Test, LargeValue) {
    // FP16 max is ~65504
    float large = 1000.0f;
    uint16_t fp16 = milvus::F32toF16(large);
    float result = milvus::F16toF32(fp16);
    EXPECT_NEAR(result, large, 1.0f);

    // BF16 supports much larger range
    large = 1e10f;
    uint16_t bf16 = milvus::F32toBF16(large);
    result = milvus::BF16toF32(bf16);
    EXPECT_NEAR(result, large, large * 0.01f);
}

TEST_F(FP16Test, EmptyArrays) {
    std::vector<float> empty;

    auto fp16_result = milvus::ArrayF32toF16(empty);
    EXPECT_TRUE(fp16_result.empty());

    auto f32_result = milvus::ArrayF16toF32(std::vector<uint16_t>{});
    EXPECT_TRUE(f32_result.empty());

    auto bf16_result = milvus::ArrayF32toBF16(empty);
    EXPECT_TRUE(bf16_result.empty());

    auto f32_result2 = milvus::ArrayBF16toF32(std::vector<uint16_t>{});
    EXPECT_TRUE(f32_result2.empty());
}
