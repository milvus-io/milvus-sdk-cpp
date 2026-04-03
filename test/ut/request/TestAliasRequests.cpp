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

class CreateAliasRequestTest : public ::testing::Test {};

TEST_F(CreateAliasRequestTest, GettersAndSetters) {
    milvus::CreateAliasRequest req;

    req.WithCollectionName("alias_coll");
    EXPECT_EQ(req.CollectionName(), "alias_coll");

    req.WithAlias("my_alias");
    EXPECT_EQ(req.Alias(), "my_alias");

    req.WithDatabaseName("alias_db");
    EXPECT_EQ(req.DatabaseName(), "alias_db");
}

class AlterAliasRequestTest : public ::testing::Test {};

TEST_F(AlterAliasRequestTest, GettersAndSetters) {
    milvus::AlterAliasRequest req;

    req.WithCollectionName("alter_alias_coll");
    EXPECT_EQ(req.CollectionName(), "alter_alias_coll");

    req.WithAlias("altered_alias");
    EXPECT_EQ(req.Alias(), "altered_alias");
}

class DropAliasRequestTest : public ::testing::Test {};

TEST_F(DropAliasRequestTest, GettersAndSetters) {
    milvus::DropAliasRequest req;

    req.WithAlias("drop_alias");
    EXPECT_EQ(req.Alias(), "drop_alias");

    req.WithDatabaseName("drop_alias_db");
    EXPECT_EQ(req.DatabaseName(), "drop_alias_db");
}

class DescribeAliasRequestTest : public ::testing::Test {};

TEST_F(DescribeAliasRequestTest, GettersAndSetters) {
    milvus::DescribeAliasRequest req;

    req.WithAlias("desc_alias");
    EXPECT_EQ(req.Alias(), "desc_alias");

    req.WithDatabaseName("desc_alias_db");
    EXPECT_EQ(req.DatabaseName(), "desc_alias_db");
}

class ListAliasesRequestTest : public ::testing::Test {};

TEST_F(ListAliasesRequestTest, GettersAndSetters) {
    milvus::ListAliasesRequest req;

    req.WithCollectionName("list_alias_coll");
    EXPECT_EQ(req.CollectionName(), "list_alias_coll");

    req.WithDatabaseName("list_alias_db");
    EXPECT_EQ(req.DatabaseName(), "list_alias_db");
}

TEST_F(ListAliasesRequestTest, SetDatabaseNameAndCollectionName) {
    milvus::ListAliasesRequest req;
    req.SetDatabaseName("db_set");
    EXPECT_EQ(req.DatabaseName(), "db_set");

    req.SetCollectionName("coll_set");
    EXPECT_EQ(req.CollectionName(), "coll_set");
}
