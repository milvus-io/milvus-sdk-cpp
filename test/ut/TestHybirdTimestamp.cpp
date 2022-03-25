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

#include "milvus/types/HybirdTimestamp.h"

class HybirdTimestampTest : public ::testing::Test {};

TEST_F(HybirdTimestampTest, GeneralTesting) {
    milvus::HybirdTimestamp foo{};
    EXPECT_EQ(foo.Timestamp(), 0);
    EXPECT_EQ(foo.Logical(), 0);
    EXPECT_EQ(foo.Physical(), 0);

    milvus::HybirdTimestamp bar{1, 1};
    EXPECT_EQ(bar.Timestamp(), (1 << 18) + 1);
    EXPECT_EQ(bar.Logical(), 1);
    EXPECT_EQ(bar.Physical(), 1);

    foo += 1ULL;
    EXPECT_EQ(foo.Logical(), 0);
    EXPECT_EQ(foo.Physical(), 1);

    foo += std::chrono::seconds{1};
    EXPECT_EQ(foo.Logical(), 0);
    EXPECT_EQ(foo.Physical(), 1001);

    bar = foo + std::chrono::seconds{1};
    EXPECT_EQ(bar.Logical(), 0);
    EXPECT_EQ(bar.Physical(), 2001);

    bar = milvus::HybirdTimestamp::CreateFromUnixtime(1000);
    EXPECT_EQ(bar.Logical(), 0);
    EXPECT_EQ(bar.Physical(), 1000);

    milvus::HybirdTimestamp ts_a = milvus::HybirdTimestamp::CreateFromUnixtime(3);
    milvus::HybirdTimestamp ts_b{ts_a.Timestamp()};
    EXPECT_EQ(ts_a.Logical(), ts_b.Logical());
    EXPECT_EQ(ts_a.Physical(), ts_b.Physical());
}
