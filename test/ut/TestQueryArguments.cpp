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

#include "milvus/types/QueryArguments.h"

class QueryArgumentsTest : public ::testing::Test {};

TEST_F(QueryArgumentsTest, GeneralTesting) {
    milvus::QueryArguments arguments;

    std::string empty_name = "";
    std::string collection_name = "test";
    arguments.SetCollectionName(collection_name);
    EXPECT_FALSE(arguments.SetCollectionName(empty_name).IsOk());
    EXPECT_EQ(collection_name, arguments.CollectionName());

    std::string partition_name = "p1";
    arguments.AddPartitionName(partition_name);
    EXPECT_FALSE(arguments.AddPartitionName(empty_name).IsOk());
    EXPECT_EQ(1, arguments.PartitionNames().size());
    auto names = arguments.PartitionNames();
    EXPECT_TRUE(names.find(partition_name) != names.end());

    std::string field_name = "f1";
    arguments.AddOutputField(field_name);
    EXPECT_FALSE(arguments.AddOutputField(empty_name).IsOk());
    EXPECT_EQ(1, arguments.OutputFields().size());
    auto field_names = arguments.OutputFields();
    EXPECT_TRUE(field_names.find(field_name) != field_names.end());

    std::string expression = "expr";
    arguments.SetExpression(expression);
    EXPECT_FALSE(arguments.SetExpression(empty_name).IsOk());
    EXPECT_EQ(expression, arguments.Expression());

    uint64_t ts = 1000;
    arguments.SetTravelTimestamp(ts);
    EXPECT_EQ(ts, arguments.TravelTimestamp());
    arguments.SetGuaranteeTimestamp(ts);
    EXPECT_EQ(ts, arguments.GuaranteeTimestamp());
}
