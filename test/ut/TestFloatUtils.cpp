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

#include "Eigen/Core"
#include "milvus/types/FloatUtils.h"

class FloatUtilsTest : public ::testing::Test {
 protected:
    template <typename Fp16T, typename FloatT>
    static void
    FloatConversion() {
        std::vector<FloatT> data;
        for (size_t i = 0; i < 100; ++i) {
            data.push_back(FloatT(static_cast<float>(i) * 0.111));
        }
        std::string bytes = milvus::FloatNumVecToFloat16NumVecBytes<FloatT, Fp16T>(data);
        ASSERT_EQ(bytes.size(), 100 * 2);
        std::vector<FloatT> result = milvus::Float16NumVecBytesToFloatNumVec<Fp16T, FloatT>(bytes);
        ASSERT_EQ(data.size(), result.size());
        for (size_t i = 0; i < data.size(); ++i) {
            ASSERT_NEAR(data[i], result[i], 4e-2);
        }
    }
};

TEST_F(FloatUtilsTest, Float16NumVecBytesToFloatNumVec) {
    FloatConversion<Eigen::half, Eigen::half>();
    FloatConversion<Eigen::half, float>();
    FloatConversion<Eigen::half, double>();
    FloatConversion<Eigen::bfloat16, Eigen::bfloat16>();
    FloatConversion<Eigen::bfloat16, float>();
    FloatConversion<Eigen::bfloat16, double>();
}
