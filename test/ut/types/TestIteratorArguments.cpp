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

class QueryIteratorArgumentsTest : public ::testing::Test {};

TEST_F(QueryIteratorArgumentsTest, DefaultBatchSize) {
    milvus::QueryIteratorArguments args;
    EXPECT_EQ(args.BatchSize(), 1000);
}

TEST_F(QueryIteratorArgumentsTest, SetBatchSize) {
    milvus::QueryIteratorArguments args;
    auto status = args.SetBatchSize(500);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(args.BatchSize(), 500);
}

TEST_F(QueryIteratorArgumentsTest, DefaultCollectionID) {
    milvus::QueryIteratorArguments args;
    EXPECT_EQ(args.CollectionID(), 0);
}

TEST_F(QueryIteratorArgumentsTest, SetCollectionID) {
    milvus::QueryIteratorArguments args;
    auto status = args.SetCollectionID(12345);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(args.CollectionID(), 12345);
}

TEST_F(QueryIteratorArgumentsTest, SetPkSchema) {
    milvus::QueryIteratorArguments args;
    milvus::FieldSchema schema("pk", milvus::DataType::INT64, "", true);
    auto status = args.SetPkSchema(schema);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(args.PkSchema().Name(), "pk");
}

TEST_F(QueryIteratorArgumentsTest, ReduceStopForBest) {
    milvus::QueryIteratorArguments args;
    EXPECT_FALSE(args.ReduceStopForBest());

    auto status = args.SetReduceStopForBest(true);
    EXPECT_TRUE(status.IsOk());
    EXPECT_TRUE(args.ReduceStopForBest());
}

class SearchIteratorArgumentsTest : public ::testing::Test {};

TEST_F(SearchIteratorArgumentsTest, DefaultBatchSize) {
    milvus::SearchIteratorArguments args;
    EXPECT_EQ(args.BatchSize(), 1000);
}

TEST_F(SearchIteratorArgumentsTest, SetBatchSize) {
    milvus::SearchIteratorArguments args;
    auto status = args.SetBatchSize(200);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(args.BatchSize(), 200);
}

TEST_F(SearchIteratorArgumentsTest, SetCollectionID) {
    milvus::SearchIteratorArguments args;
    auto status = args.SetCollectionID(999);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(args.CollectionID(), 999);
}
