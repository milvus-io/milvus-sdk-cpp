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

#include <gmock/gmock.h>

#include "utils/GtsDict.h"

class GtsDictTest : public ::testing::Test {};

TEST_F(GtsDictTest, GeneralTest) {
    milvus::GtsDict& instance = milvus::GtsDict::GetInstance();
    instance.UpdateCollectionTs("db", "aaa", 1000);

    // "bbb" doesn't exist, return false
    uint64_t ts = 0;
    auto ret = instance.GetCollectionTs("db", "bbb", ts);
    EXPECT_FALSE(ret);

    // get correct ts of "aaa"
    ret = instance.GetCollectionTs("db", "aaa", ts);
    EXPECT_TRUE(ret);
    EXPECT_EQ(1000, ts);

    // "bbb" doesn't exist, do nothing
    instance.RemoveCollectionTs("db", "bbb");

    // add ts for "bbb"
    instance.UpdateCollectionTs("db", "bbb", 999);
    ret = instance.GetCollectionTs("db", "bbb", ts);
    EXPECT_TRUE(ret);
    EXPECT_EQ(999, ts);

    // remove ts of "aaa"
    instance.RemoveCollectionTs("db", "aaa");
    ret = instance.GetCollectionTs("db", "aaa", ts);
    EXPECT_FALSE(ret);

    // remove all
    instance.CleanAllCollectionTs();
    ret = instance.GetCollectionTs("db", "bbb", ts);
    EXPECT_FALSE(ret);
}