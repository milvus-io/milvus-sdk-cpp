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

#include "milvus/types/CalcDistanceArguments.h"

class CalcDistanceArgumentsTest : public ::testing::Test {};

TEST_F(CalcDistanceArgumentsTest, GeneralTesting) {
    milvus::CalcDistanceArguments arguments;
    auto status = arguments.SetMetricType("invalid");
    EXPECT_FALSE(status.IsOk());

    EXPECT_EQ("L2", arguments.MetricType());

    status = arguments.SetMetricType("l2");
    EXPECT_TRUE(status.IsOk());

    status = arguments.SetMetricType("Ip");
    EXPECT_TRUE(status.IsOk());

    status = arguments.SetMetricType("Hamming");
    EXPECT_TRUE(status.IsOk());

    status = arguments.SetMetricType("TANIMOTo");
    EXPECT_TRUE(status.IsOk());

    auto metric = arguments.MetricType();
    EXPECT_EQ("TANIMOTO", arguments.MetricType());

    EXPECT_EQ(0, arguments.Dimension());

    status = arguments.SetDimension(0);
    EXPECT_FALSE(status.IsOk());

    int32_t dim = 31;
    status = arguments.SetDimension(dim);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(dim, arguments.Dimension());

    EXPECT_EQ(false, arguments.Sqrt());
    bool sqrt = true;
    arguments.SetSqrt(sqrt);
    EXPECT_EQ(sqrt, arguments.Sqrt());
}

TEST_F(CalcDistanceArgumentsTest, FloatVectors) {
    milvus::FloatVecFieldDataPtr vectors_1 = std::make_shared<milvus::FloatVecFieldData>();
    milvus::FloatVecFieldData::ElementT element_1 = {1.0, 2.0};
    milvus::FloatVecFieldData::ElementT element_2 = {3.0, 4.0};
    vectors_1->Add(element_1);
    vectors_1->Add(element_2);

    milvus::FloatVecFieldDataPtr vectors_2 = std::make_shared<milvus::FloatVecFieldData>();
    milvus::FloatVecFieldData::ElementT element_3 = {1.0, 2.0};
    milvus::FloatVecFieldData::ElementT element_4 = {3.0, 4.0};
    milvus::FloatVecFieldData::ElementT element_5 = {5.0, 6.0};
    vectors_2->Add(element_3);
    vectors_2->Add(element_4);
    vectors_2->Add(element_5);

    milvus::CalcDistanceArguments arguments;
    milvus::FloatVecFieldDataPtr vectors_3 = nullptr;
    auto status = arguments.SetLeftVectors(std::move(vectors_3));
    EXPECT_FALSE(status.IsOk());
    status = arguments.SetRightVectors(std::move(vectors_3));
    EXPECT_FALSE(status.IsOk());

    arguments.SetLeftVectors(std::move(vectors_1));
    arguments.SetRightVectors(std::move(vectors_2));

    EXPECT_NE(arguments.LeftVectors(), nullptr);
    EXPECT_NE(arguments.RightVectors(), nullptr);

    EXPECT_EQ(arguments.LeftVectors()->Count(), 2);
    EXPECT_EQ(arguments.RightVectors()->Count(), 3);
}

TEST_F(CalcDistanceArgumentsTest, BinaryVectors) {
    milvus::BinaryVecFieldDataPtr vectors_1 = std::make_shared<milvus::BinaryVecFieldData>();
    milvus::BinaryVecFieldData::ElementT element_1 = {1, 2};
    milvus::BinaryVecFieldData::ElementT element_2 = {3, 4};
    vectors_1->Add(element_1);
    vectors_1->Add(element_2);

    milvus::BinaryVecFieldDataPtr vectors_2 = std::make_shared<milvus::BinaryVecFieldData>();
    milvus::BinaryVecFieldData::ElementT element_3 = {5, 6};
    milvus::BinaryVecFieldData::ElementT element_4 = {7, 8};
    milvus::BinaryVecFieldData::ElementT element_5 = {9, 10};
    vectors_2->Add(element_3);
    vectors_2->Add(element_4);
    vectors_2->Add(element_5);

    milvus::CalcDistanceArguments arguments;
    milvus::BinaryVecFieldDataPtr vectors_3 = nullptr;
    auto status = arguments.SetLeftVectors(std::move(vectors_3));
    EXPECT_FALSE(status.IsOk());
    status = arguments.SetRightVectors(std::move(vectors_3));
    EXPECT_FALSE(status.IsOk());

    arguments.SetLeftVectors(std::move(vectors_1));
    arguments.SetRightVectors(std::move(vectors_2));

    EXPECT_NE(arguments.LeftVectors(), nullptr);
    EXPECT_NE(arguments.RightVectors(), nullptr);

    EXPECT_EQ(arguments.LeftVectors()->Count(), 2);
    EXPECT_EQ(arguments.RightVectors()->Count(), 3);
}

TEST_F(CalcDistanceArgumentsTest, VectorIDs) {
    std::vector<int64_t> ids_1 = {1, 2};
    std::vector<int64_t> ids_2 = {3, 4, 5};

    std::string left_collection = "c1";
    std::string right_collection = "c1";
    std::vector<std::string> left_partitions = {"p1", "p2"};
    std::vector<std::string> right_partitions = {"p3"};

    milvus::CalcDistanceArguments arguments;

    auto status = arguments.SetLeftVectors(std::make_shared<milvus::Int64FieldData>("", ids_1), left_collection);
    EXPECT_FALSE(status.IsOk());

    arguments.SetRightVectors(std::make_shared<milvus::Int64FieldData>("", ids_2), right_collection);
    EXPECT_FALSE(status.IsOk());

    status = arguments.SetLeftVectors(std::make_shared<milvus::Int64FieldData>("f1", std::vector<int64_t>{}),
                                      left_collection);
    EXPECT_FALSE(status.IsOk());

    arguments.SetRightVectors(std::make_shared<milvus::Int64FieldData>("f2", std::vector<int64_t>{}), right_collection);
    EXPECT_FALSE(status.IsOk());

    status = arguments.SetLeftVectors(std::make_shared<milvus::Int64FieldData>("f1", ids_1), left_collection,
                                      left_partitions);
    EXPECT_TRUE(status.IsOk());

    status = arguments.SetRightVectors(std::make_shared<milvus::Int64FieldData>("f2", ids_2), right_collection,
                                       right_partitions);
    EXPECT_TRUE(status.IsOk());

    status =
        arguments.SetLeftVectors(std::make_shared<milvus::Int64FieldData>("f1", ids_1), "", std::move(left_partitions));
    EXPECT_FALSE(status.IsOk());

    status = arguments.SetRightVectors(std::make_shared<milvus::Int64FieldData>("f2", ids_2), "",
                                       std::move(right_partitions));
    EXPECT_FALSE(status.IsOk());

    EXPECT_NE(arguments.LeftVectors(), nullptr);
    EXPECT_NE(arguments.RightVectors(), nullptr);

    EXPECT_EQ(arguments.LeftVectors()->Count(), 2);
    EXPECT_EQ(arguments.RightVectors()->Count(), 3);

    EXPECT_EQ(arguments.LeftCollection(), left_collection);
    EXPECT_EQ(arguments.RightCollection(), right_collection);

    EXPECT_EQ(arguments.LeftPartitions().size(), 2);
    EXPECT_EQ(arguments.RightPartitions().size(), 1);
}

TEST_F(CalcDistanceArgumentsTest, Validate) {
    milvus::FloatVecFieldDataPtr vectors_1 = std::make_shared<milvus::FloatVecFieldData>();
    milvus::FloatVecFieldData::ElementT element_1 = {1.0, 2.0};
    milvus::FloatVecFieldData::ElementT element_2 = {3.0, 4.0};
    vectors_1->Add(element_1);
    vectors_1->Add(element_2);

    milvus::BinaryVecFieldDataPtr vectors_2 = std::make_shared<milvus::BinaryVecFieldData>();
    milvus::BinaryVecFieldData::ElementT element_3 = {5, 6};
    milvus::BinaryVecFieldData::ElementT element_4 = {7, 8};
    milvus::BinaryVecFieldData::ElementT element_5 = {9, 10};
    vectors_2->Add(element_3);
    vectors_2->Add(element_4);
    vectors_2->Add(element_5);

    milvus::CalcDistanceArguments arguments;
    auto status = arguments.Validate();
    EXPECT_FALSE(status.IsOk());

    arguments.SetLeftVectors(std::move(vectors_1));
    status = arguments.Validate();
    EXPECT_FALSE(status.IsOk());

    arguments.SetRightVectors(milvus::BinaryVecFieldDataPtr{vectors_2});
    status = arguments.Validate();
    EXPECT_FALSE(status.IsOk());

    milvus::FloatVecFieldDataPtr vectors_3 = std::make_shared<milvus::FloatVecFieldData>();
    milvus::FloatVecFieldData::ElementT element_6 = {1.0, 2.0};
    vectors_3->Add(element_6);

    arguments.SetRightVectors(std::move(vectors_3));
    status = arguments.Validate();
    EXPECT_TRUE(status.IsOk());

    arguments.SetMetricType("Hamming");
    status = arguments.Validate();
    EXPECT_FALSE(status.IsOk());

    milvus::BinaryVecFieldDataPtr vectors_4 = std::make_shared<milvus::BinaryVecFieldData>();
    milvus::BinaryVecFieldData::ElementT element_7 = {1, 2};
    vectors_4->Add(element_7);

    arguments.SetLeftVectors(milvus::BinaryVecFieldDataPtr{vectors_2});
    arguments.SetRightVectors(std::move(vectors_4));
    arguments.SetMetricType("L2");
    status = arguments.Validate();
    EXPECT_FALSE(status.IsOk());
}
