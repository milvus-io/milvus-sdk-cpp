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

class PartitionInfoTest : public ::testing::Test {};

TEST_F(PartitionInfoTest, ConstructorWithAllParams) {
    milvus::PartitionInfo info("part_1", 100, 1234567890);
    EXPECT_EQ(info.Name(), "part_1");
    EXPECT_EQ(info.Id(), 100);
    EXPECT_EQ(info.CreatedUtcTimestamp(), 1234567890u);
}

TEST_F(PartitionInfoTest, ConstructorWithDefaultTimestamp) {
    milvus::PartitionInfo info("part_2", 200);
    EXPECT_EQ(info.Name(), "part_2");
    EXPECT_EQ(info.Id(), 200);
    EXPECT_EQ(info.CreatedUtcTimestamp(), 0u);
}

TEST_F(PartitionInfoTest, DeprecatedMethods) {
    milvus::PartitionInfo info("part_3", 300, 999);
    EXPECT_EQ(info.InMemoryPercentage(), 0);
    EXPECT_FALSE(info.Loaded());
}

TEST_F(PartitionInfoTest, EqualityOperator) {
    milvus::PartitionInfo a("part", 1, 100);
    milvus::PartitionInfo b("part", 1, 100);
    EXPECT_TRUE(a == b);

    milvus::PartitionInfo c("other", 2, 200);
    EXPECT_FALSE(a == c);
}
