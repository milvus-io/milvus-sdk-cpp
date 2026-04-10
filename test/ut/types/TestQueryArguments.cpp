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

    std::string empty_name;
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
    arguments.SetFilter(expression);
    EXPECT_FALSE(arguments.SetFilter(empty_name).IsOk());
    EXPECT_EQ(expression, arguments.Filter());

    uint64_t ts = 1000;
    arguments.SetTravelTimestamp(ts);
    EXPECT_EQ(ts, arguments.TravelTimestamp());

    arguments.SetLimit(88);
    EXPECT_EQ(88, arguments.Limit());
    arguments.SetOffset(99);
    EXPECT_EQ(99, arguments.Offset());
}

TEST_F(QueryArgumentsTest, DatabaseName) {
    milvus::QueryArguments arguments;
    EXPECT_EQ("", arguments.DatabaseName());

    arguments.SetDatabaseName("mydb");
    EXPECT_EQ("mydb", arguments.DatabaseName());
}

TEST_F(QueryArgumentsTest, FilterTemplates) {
    milvus::QueryArguments arguments;
    EXPECT_TRUE(arguments.FilterTemplates().empty());

    nlohmann::json val = 42;
    auto status = arguments.AddFilterTemplate("age", val);
    EXPECT_TRUE(status.IsOk());

    nlohmann::json cities = nlohmann::json::array({"beijing", "shanghai"});
    status = arguments.AddFilterTemplate("city", cities);
    EXPECT_TRUE(status.IsOk());

    EXPECT_EQ(arguments.FilterTemplates().size(), 2);
    EXPECT_EQ(arguments.FilterTemplates().at("age"), 42);
    EXPECT_EQ(arguments.FilterTemplates().at("city"), cities);
}

TEST_F(QueryArgumentsTest, GuaranteeTimestamp) {
    milvus::QueryArguments arguments;
    EXPECT_EQ(0, arguments.GuaranteeTimestamp());

    // SetGuaranteeTimestamp is deprecated and always returns OK but value is not stored
    auto status = arguments.SetGuaranteeTimestamp(5000);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(QueryArgumentsTest, IgnoreGrowing) {
    milvus::QueryArguments arguments;
    EXPECT_FALSE(arguments.IgnoreGrowing());

    arguments.SetIgnoreGrowing(true);
    EXPECT_TRUE(arguments.IgnoreGrowing());

    arguments.SetIgnoreGrowing(false);
    EXPECT_FALSE(arguments.IgnoreGrowing());
}

TEST_F(QueryArgumentsTest, ExtraParams) {
    milvus::QueryArguments arguments;
    EXPECT_TRUE(arguments.ExtraParams().empty());

    auto status = arguments.AddExtraParam("key1", "val1");
    EXPECT_TRUE(status.IsOk());

    EXPECT_EQ(arguments.ExtraParams().size(), 1);
    EXPECT_EQ(arguments.ExtraParams().at("key1"), "val1");
}

TEST_F(QueryArgumentsTest, ConsistencyLevel) {
    milvus::QueryArguments arguments;
    EXPECT_EQ(milvus::ConsistencyLevel::NONE, arguments.GetConsistencyLevel());

    arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    EXPECT_EQ(milvus::ConsistencyLevel::STRONG, arguments.GetConsistencyLevel());

    arguments.SetConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);
    EXPECT_EQ(milvus::ConsistencyLevel::EVENTUALLY, arguments.GetConsistencyLevel());
}
