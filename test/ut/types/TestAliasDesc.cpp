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

class AliasDescTest : public ::testing::Test {};

TEST_F(AliasDescTest, DefaultConstructor) {
    milvus::AliasDesc desc;
    EXPECT_TRUE(desc.Name().empty());
    EXPECT_TRUE(desc.DatabaseName().empty());
    EXPECT_TRUE(desc.CollectionName().empty());
}

TEST_F(AliasDescTest, ParameterizedConstructor) {
    milvus::AliasDesc desc("my_alias", "my_db", "my_collection");
    EXPECT_EQ(desc.Name(), "my_alias");
    EXPECT_EQ(desc.DatabaseName(), "my_db");
    EXPECT_EQ(desc.CollectionName(), "my_collection");
}

TEST_F(AliasDescTest, SettersAndGetters) {
    milvus::AliasDesc desc;

    desc.SetName("alias_1");
    EXPECT_EQ(desc.Name(), "alias_1");

    desc.SetDatabaseName("db_1");
    EXPECT_EQ(desc.DatabaseName(), "db_1");

    desc.SetCollectionName("coll_1");
    EXPECT_EQ(desc.CollectionName(), "coll_1");
}
