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

class DescribeAliasResponseTest : public ::testing::Test {};

TEST_F(DescribeAliasResponseTest, SetterAndGetter) {
    milvus::DescribeAliasResponse resp;
    milvus::AliasDesc desc;
    resp.SetDesc(std::move(desc));
    // Just verify it does not crash
    (void)resp.Desc();
}

class ListAliasesResponseTest : public ::testing::Test {};

TEST_F(ListAliasesResponseTest, SetterAndGetter) {
    milvus::ListAliasesResponse resp;

    resp.SetDatabaseName("db1");
    EXPECT_EQ(resp.DatabaseName(), "db1");

    resp.SetCollectionName("coll1");
    EXPECT_EQ(resp.CollectionName(), "coll1");

    std::vector<std::string> aliases{"a1", "a2"};
    resp.SetAliases(std::move(aliases));
    EXPECT_EQ(resp.Aliases().size(), 2);
}
