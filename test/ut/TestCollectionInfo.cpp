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

#include "milvus/types/CollectionInfo.h"

class CollectionInfoTest : public ::testing::Test {};

TEST_F(CollectionInfoTest, GeneralTesting) {
    std::string name = "test";
    int64_t id = 1;
    uint64_t created_ts = 100;
    uint64_t in_memory_percent = 50;

    milvus::CollectionInfo info(name, id, created_ts, in_memory_percent);
    EXPECT_EQ(name, info.Name());
    EXPECT_EQ(id, info.ID());
    EXPECT_EQ(created_ts, info.CreatedTime());
    EXPECT_EQ(in_memory_percent, info.MemoryPercentage());

    milvus::CollectionInfo info_bar;
    EXPECT_EQ("", info_bar.Name());
    EXPECT_EQ(0, info_bar.ID());
    EXPECT_EQ(0, info_bar.CreatedTime());
    EXPECT_EQ(0, info_bar.MemoryPercentage());
}
