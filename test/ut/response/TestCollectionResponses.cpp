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

class HasCollectionResponseTest : public ::testing::Test {};

TEST_F(HasCollectionResponseTest, SetterAndGetter) {
    milvus::HasCollectionResponse resp;
    EXPECT_FALSE(resp.Has());

    resp.SetHas(true);
    EXPECT_TRUE(resp.Has());

    resp.SetHas(false);
    EXPECT_FALSE(resp.Has());
}

class ListCollectionsResponseTest : public ::testing::Test {};

TEST_F(ListCollectionsResponseTest, SetterAndGetter) {
    milvus::ListCollectionsResponse resp;

    std::vector<std::string> names{"coll1", "coll2"};
    resp.SetCollectionNames(std::move(names));
    EXPECT_EQ(resp.CollectionNames().size(), 2);
    EXPECT_EQ(resp.CollectionNames()[0], "coll1");

    std::vector<milvus::CollectionInfo> infos;
    milvus::CollectionInfo info;
    infos.push_back(info);
    resp.SetCollectionInfos(std::move(infos));
    EXPECT_EQ(resp.CollectionInfos().size(), 1);
}

class DescribeCollectionResponseTest : public ::testing::Test {};

TEST_F(DescribeCollectionResponseTest, SetterAndGetter) {
    milvus::DescribeCollectionResponse resp;
    milvus::CollectionDesc desc;
    resp.SetDesc(std::move(desc));
    (void)resp.Desc();
}

class GetCollectionStatsResponseTest : public ::testing::Test {};

TEST_F(GetCollectionStatsResponseTest, SetterAndGetter) {
    milvus::GetCollectionStatsResponse resp;
    milvus::CollectionStat stats;
    resp.SetStats(std::move(stats));
    (void)resp.Stats();
}

class GetLoadStateResponseTest : public ::testing::Test {};

TEST_F(GetLoadStateResponseTest, SetterAndGetter) {
    milvus::GetLoadStateResponse resp;

    resp.SetState(milvus::LoadState::LOAD_STATE_LOADED);
    EXPECT_EQ(resp.State(), milvus::LoadState::LOAD_STATE_LOADED);

    resp.SetProgress(100);
    EXPECT_EQ(resp.Progress(), 100);
}
