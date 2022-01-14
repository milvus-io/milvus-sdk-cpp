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

#include "milvus/types/FieldData.h"

class FieldDataTest : public ::testing::Test {};

TEST_F(FieldDataTest, GeneralTesting) {
    std::string name = "dummy";

    {
        milvus::BoolFieldData data{name};
        data.Add(true);
        data.Add(false);
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::BOOL);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0), true);
        EXPECT_EQ(data.Data().at(1), false);

        std::vector<bool> values = {true, false};
        milvus::BoolFieldData cp{name, values};
        EXPECT_EQ(cp.Data().size(), 2);
    }

    {
        milvus::Int8FieldData data{name};
        data.Add(1);
        data.Add(2);
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::INT8);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0), 1);
        EXPECT_EQ(data.Data().at(1), 2);

        std::vector<int8_t> values = {1, 2};
        milvus::Int8FieldData cp{name, values};
        EXPECT_EQ(cp.Data().size(), 2);
    }

    {
        milvus::Int16FieldData data{name};
        data.Add(1);
        data.Add(2);
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::INT16);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0), 1);
        EXPECT_EQ(data.Data().at(1), 2);

        std::vector<int16_t> values = {1, 2};
        milvus::Int16FieldData cp{name, values};
        EXPECT_EQ(cp.Data().size(), 2);
    }

    {
        milvus::Int32FieldData data{name};
        data.Add(1);
        data.Add(2);
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::INT32);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0), 1);
        EXPECT_EQ(data.Data().at(1), 2);

        std::vector<int32_t> values = {1, 2};
        milvus::Int32FieldData cp{name, values};
        EXPECT_EQ(cp.Data().size(), 2);
    }

    {
        milvus::Int64FieldData data{name};
        data.Add(1);
        data.Add(2);
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::INT64);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0), 1);
        EXPECT_EQ(data.Data().at(1), 2);

        std::vector<int64_t> values = {1, 2};
        milvus::Int64FieldData cp{name, values};
        EXPECT_EQ(cp.Data().size(), 2);
    }

    {
        milvus::FloatFieldData data{name};
        data.Add(1.1);
        data.Add(2.2);
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::FLOAT);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_FLOAT_EQ(data.Data().at(0), 1.1);
        EXPECT_FLOAT_EQ(data.Data().at(1), 2.2);

        std::vector<float> values = {1.0, 2.0};
        milvus::FloatFieldData cp{name, values};
        EXPECT_EQ(cp.Data().size(), 2);
    }

    {
        milvus::DoubleFieldData data{name};
        data.Add(1.1);
        data.Add(2.2);
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::DOUBLE);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_DOUBLE_EQ(data.Data().at(0), 1.1);
        EXPECT_DOUBLE_EQ(data.Data().at(1), 2.2);

        std::vector<double> values = {1.0, 2.0};
        milvus::DoubleFieldData cp{name, values};
        EXPECT_EQ(cp.Data().size(), 2);
    }

    {
        milvus::StringFieldData data{name};
        data.Add("a");
        data.Add("b");
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::STRING);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0), "a");
        EXPECT_EQ(data.Data().at(1), "b");

        std::vector<std::string> values = {"a", "b"};
        milvus::StringFieldData cp{name, values};
        EXPECT_EQ(cp.Data().size(), 2);
    }

    {
        milvus::BinaryVecFieldData data{name};
        milvus::BinaryVecFieldData::ElementT element_1 = {1, 2};
        milvus::BinaryVecFieldData::ElementT element_2 = {3, 4};
        data.Add(element_1);
        data.Add(element_2);
        milvus::BinaryVecFieldData::ElementT element_3 = {5};
        EXPECT_EQ(data.Add(element_3), milvus::StatusCode::DIMENSION_NOT_EQUAL);
        milvus::BinaryVecFieldData::ElementT element_4;
        EXPECT_EQ(data.Add(element_4), milvus::StatusCode::VECTOR_IS_EMPTY);
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::BINARY_VECTOR);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_FLOAT_EQ(data.Data().at(0).size(), element_1.size());
        EXPECT_FLOAT_EQ(data.Data().at(1).size(), element_2.size());

        std::vector<milvus::BinaryVecFieldData::ElementT> values = {element_1, element_2};
        milvus::BinaryVecFieldData cp{name, values};
        EXPECT_EQ(cp.Data().size(), 2);
    }

    {
        milvus::FloatVecFieldData data{name};
        milvus::FloatVecFieldData::ElementT element_1 = {1.0, 2.0};
        milvus::FloatVecFieldData::ElementT element_2 = {3.0, 4.0};
        data.Add(element_1);
        data.Add(element_2);
        milvus::FloatVecFieldData::ElementT element_3 = {5.0};
        EXPECT_EQ(data.Add(element_3), milvus::StatusCode::DIMENSION_NOT_EQUAL);
        milvus::FloatVecFieldData::ElementT element_4;
        EXPECT_EQ(data.Add(element_4), milvus::StatusCode::VECTOR_IS_EMPTY);
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::FLOAT_VECTOR);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_FLOAT_EQ(data.Data().at(0).size(), element_1.size());
        EXPECT_FLOAT_EQ(data.Data().at(1).size(), element_2.size());

        std::vector<milvus::FloatVecFieldData::ElementT> values = {element_1, element_2};
        milvus::FloatVecFieldData cp{name, values};
        EXPECT_EQ(cp.Data().size(), 2);
    }
}
