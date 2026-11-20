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

#include "milvus/utils/CacheConfig.h"
#include "utils/cache/CollectionTsCache.h"
#include "utils/cache/SchemaCache.h"

TEST(CacheConfigTest, SetCapacityTakesEffectOnBothCaches) {
    const auto original_schema = milvus::SchemaCache::GetInstance().Capacity();
    const auto original_ts = milvus::CollectionTsCache::GetInstance().Capacity();

    auto status = milvus::CacheConfig::SetCapacity(256);
    ASSERT_TRUE(status.IsOk());

    EXPECT_EQ(milvus::CacheConfig::GetCapacity(), 256);
    EXPECT_EQ(milvus::SchemaCache::GetInstance().Capacity(), 256);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Capacity(), 256);

    // Restore each singleton through its own SetCapacity so the process-wide
    // state is correct even if the two defaults ever differ.
    milvus::SchemaCache::GetInstance().SetCapacity(original_schema);
    milvus::CollectionTsCache::GetInstance().SetCapacity(original_ts);
}

TEST(CacheConfigTest, SetCapacityRejectsZero) {
    auto status = milvus::CacheConfig::SetCapacity(0);
    ASSERT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
}
