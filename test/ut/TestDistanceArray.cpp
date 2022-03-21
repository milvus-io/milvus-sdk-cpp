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

#include "milvus/types/DistanceArray.h"

class DistanceArrayTest : public ::testing::Test {};

TEST_F(DistanceArrayTest, GeneralTesting) {
    std::vector<std::vector<int32_t>> array_foo = {{1, 2, 3}, {2, 3, 4}};
    std::vector<std::vector<float>> array_bar = {{1.f, 2.f, 3.f}, {2.f, 3.f, 4.f}};

    milvus::DistanceArray distance_array;
    EXPECT_FALSE(distance_array.IsIntegerDistance());

    distance_array = milvus::DistanceArray(array_foo);
    EXPECT_TRUE(distance_array.IsIntegerDistance());
    EXPECT_EQ(distance_array.IntDistanceArray(), array_foo);

    distance_array = milvus::DistanceArray(array_bar);
    EXPECT_FALSE(distance_array.IsIntegerDistance());
    EXPECT_EQ(distance_array.FloatDistanceArray(), array_bar);

    distance_array.SetIntDistance(array_foo);
    EXPECT_TRUE(distance_array.IsIntegerDistance());
    EXPECT_EQ(distance_array.IntDistanceArray(), array_foo);

    distance_array.SetFloatDistance(array_bar);
    EXPECT_FALSE(distance_array.IsIntegerDistance());
    EXPECT_EQ(distance_array.FloatDistanceArray(), array_bar);

    distance_array = milvus::DistanceArray(std::move(array_foo));
    EXPECT_TRUE(distance_array.IsIntegerDistance());
    array_foo = distance_array.IntDistanceArray();

    distance_array = milvus::DistanceArray(std::move(array_bar));
    EXPECT_FALSE(distance_array.IsIntegerDistance());
    array_bar = distance_array.FloatDistanceArray();

    distance_array.SetIntDistance(std::move(array_foo));
    EXPECT_TRUE(distance_array.IsIntegerDistance());

    distance_array.SetFloatDistance(std::move(array_bar));
    EXPECT_FALSE(distance_array.IsIntegerDistance());
}
