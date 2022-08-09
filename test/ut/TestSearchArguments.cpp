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

#include "milvus/types/SearchArguments.h"

class SearchArgumentsTest : public ::testing::Test {};

TEST_F(SearchArgumentsTest, GeneralTesting) {
    milvus::SearchArguments arguments;
    EXPECT_EQ(arguments.TargetVectors(), nullptr);

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

    std::string expression = "expr";
    arguments.SetExpression(expression);
    EXPECT_EQ(expression, arguments.Expression());
    EXPECT_TRUE(arguments.SetExpression(empty_name).IsOk());

    uint64_t ts = 1000;
    arguments.SetTravelTimestamp(ts);
    EXPECT_EQ(ts, arguments.TravelTimestamp());
    arguments.SetGuaranteeTimestamp(ts);
    EXPECT_EQ(ts, arguments.GuaranteeTimestamp());

    auto status = arguments.AddOutputField("");
    EXPECT_FALSE(status.IsOk());
}

TEST_F(SearchArgumentsTest, VectorTesting) {
    std::vector<uint8_t> binary_vector = {1, 2, 3};
    std::string binary_string{'\x00', '\x00', '\x00'};
    std::vector<float> float_vector = {1.0, 2.0};

    {
        milvus::SearchArguments arguments;
        auto status = arguments.AddTargetVector("dummy", binary_vector);
        EXPECT_TRUE(status.IsOk());

        status = arguments.AddTargetVector("dummy", float_vector);
        EXPECT_FALSE(status.IsOk());

        std::vector<uint8_t> new_vector = {1, 2};
        status = arguments.AddTargetVector("dummy", new_vector);
        EXPECT_FALSE(status.IsOk());

        status = arguments.AddTargetVector("dummy", std::vector<uint8_t>{});
        EXPECT_FALSE(status.IsOk());

        auto target_vectors = arguments.TargetVectors();
        EXPECT_TRUE(target_vectors != nullptr);
        EXPECT_EQ(milvus::DataType::BINARY_VECTOR, target_vectors->Type());
        EXPECT_EQ(1, target_vectors->Count());

        status = arguments.AddTargetVector("dummy", binary_string);
        EXPECT_TRUE(status.IsOk());

        target_vectors = arguments.TargetVectors();
        EXPECT_EQ(2, target_vectors->Count());
    }

    {
        milvus::SearchArguments arguments;
        auto status = arguments.AddTargetVector("dummy", float_vector);
        EXPECT_TRUE(status.IsOk());

        status = arguments.AddTargetVector("dummy", binary_vector);
        EXPECT_FALSE(status.IsOk());

        std::vector<float> new_vector = {1.0, 2.0, 3.0};
        status = arguments.AddTargetVector("dummy", new_vector);
        EXPECT_FALSE(status.IsOk());

        status = arguments.AddTargetVector("dummy", std::vector<float>{});
        EXPECT_FALSE(status.IsOk());

        auto target_vectors = arguments.TargetVectors();
        EXPECT_TRUE(target_vectors != nullptr);
        EXPECT_EQ(milvus::DataType::FLOAT_VECTOR, target_vectors->Type());
        EXPECT_EQ(1, target_vectors->Count());
    }

    {
        milvus::SearchArguments arguments;
        auto status = arguments.AddTargetVector("dummy", std::vector<uint8_t>{1, 2, 3});
        EXPECT_TRUE(status.IsOk());

        status = arguments.AddTargetVector("dummy", std::vector<float>{1.f, 2.f});
        EXPECT_FALSE(status.IsOk());

        std::vector<uint8_t> new_vector = {1, 2};
        status = arguments.AddTargetVector("dummy", new_vector);
        EXPECT_FALSE(status.IsOk());

        status = arguments.AddTargetVector("dummy", std::vector<uint8_t>{});
        EXPECT_FALSE(status.IsOk());

        auto target_vectors = arguments.TargetVectors();
        EXPECT_TRUE(target_vectors != nullptr);
        EXPECT_EQ(milvus::DataType::BINARY_VECTOR, target_vectors->Type());
        EXPECT_EQ(1, target_vectors->Count());
    }

    {
        milvus::SearchArguments arguments;
        auto status = arguments.AddTargetVector("dummy", std::vector<float>{1.f, 2.f});
        EXPECT_TRUE(status.IsOk());

        status = arguments.AddTargetVector("dummy", std::vector<uint8_t>{1, 2, 3});
        EXPECT_FALSE(status.IsOk());

        std::vector<float> new_vector = {1.0, 2.0, 3.0};
        status = arguments.AddTargetVector("dummy", new_vector);
        EXPECT_FALSE(status.IsOk());

        status = arguments.AddTargetVector("dummy", std::vector<float>{});
        EXPECT_FALSE(status.IsOk());

        auto target_vectors = arguments.TargetVectors();
        EXPECT_TRUE(target_vectors != nullptr);
        EXPECT_EQ(milvus::DataType::FLOAT_VECTOR, target_vectors->Type());
        EXPECT_EQ(1, target_vectors->Count());
    }
}

TEST_F(SearchArgumentsTest, ValidateTesting) {
    {
        milvus::SearchArguments arguments;
        arguments.AddExtraParam("nprobe", 0);
        auto status = arguments.Validate();
        EXPECT_FALSE(status.IsOk());
    }

    {
        milvus::SearchArguments arguments;
        arguments.AddExtraParam("nprobe", 1000000);
        auto status = arguments.Validate();
        EXPECT_FALSE(status.IsOk());
    }

    {
        milvus::SearchArguments arguments;
        arguments.AddExtraParam("nprobe", 10);
        auto status = arguments.Validate();
        EXPECT_TRUE(status.IsOk());
    }
}
