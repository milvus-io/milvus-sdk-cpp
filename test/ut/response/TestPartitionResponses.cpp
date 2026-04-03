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

class HasPartitionResponseTest : public ::testing::Test {};

TEST_F(HasPartitionResponseTest, SetterAndGetter) {
    milvus::HasPartitionResponse resp;
    EXPECT_FALSE(resp.Has());

    resp.SetHas(true);
    EXPECT_TRUE(resp.Has());
}

class ListPartitionsResponseTest : public ::testing::Test {};

TEST_F(ListPartitionsResponseTest, SetterAndGetter) {
    milvus::ListPartitionsResponse resp;

    std::vector<std::string> names{"p1", "p2"};
    resp.SetPartitionNames(std::move(names));
    EXPECT_EQ(resp.PartitionsNames().size(), 2);
    EXPECT_EQ(resp.PartitionsNames()[0], "p1");

    std::vector<milvus::PartitionInfo> infos;
    infos.emplace_back("test_partition", 1, 0);
    resp.SetPartitionInfos(std::move(infos));
    EXPECT_EQ(resp.PartitionInfos().size(), 1);
}

class GetPartitionStatsResponseTest : public ::testing::Test {};

TEST_F(GetPartitionStatsResponseTest, SetterAndGetter) {
    milvus::GetPartitionStatsResponse resp;
    milvus::PartitionStat stats;
    resp.SetStats(std::move(stats));
    (void)resp.Stats();
}
