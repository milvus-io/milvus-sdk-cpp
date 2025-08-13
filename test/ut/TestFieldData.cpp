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
#include "milvus/utils/FP16.h"

class FieldDataTest : public ::testing::Test {};

namespace {
std::vector<uint16_t>
ToF16Vector(const std::vector<float>& vector, bool is_bf16) {
    std::vector<uint16_t> ret;
    ret.reserve(vector.size());
    for (auto f : vector) {
        ret.push_back(is_bf16 ? milvus::F32toBF16(f) : milvus::F32toF16(f));
    }
    return std::move(ret);
}

}  // namespace

TEST_F(FieldDataTest, ScalarFields) {
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
        milvus::BoolFieldDataPtr cp = std::make_shared<milvus::BoolFieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), false);
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
        milvus::Int8FieldDataPtr cp = std::make_shared<milvus::Int8FieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), 2);
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
        milvus::Int16FieldDataPtr cp = std::make_shared<milvus::Int16FieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), 2);
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
        milvus::Int32FieldDataPtr cp = std::make_shared<milvus::Int32FieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), 2);
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
        milvus::Int64FieldDataPtr cp = std::make_shared<milvus::Int64FieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), 2);
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
        milvus::FloatFieldDataPtr cp = std::make_shared<milvus::FloatFieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), 2.0);
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
        milvus::DoubleFieldDataPtr cp = std::make_shared<milvus::DoubleFieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), 2.0);
    }

    {
        milvus::VarCharFieldData data{name};
        data.Add("a");
        data.Add("b");
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::VARCHAR);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0), "a");
        EXPECT_EQ(data.Data().at(1), "b");

        std::vector<std::string> values = {"aa", "bb"};
        milvus::VarCharFieldDataPtr cp = std::make_shared<milvus::VarCharFieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), "bb");
    }

    {
        auto j1 = R"({"name":"aaa","age":18,"score":88})";
        auto j2 = R"({"name":"bbb","age":20,"score":99})";
        milvus::JSONFieldData data{name};
        data.Add(j1);
        data.Add(j2);
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::JSON);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0), j1);
        EXPECT_EQ(data.Data().at(1), j2);

        std::vector<nlohmann::json> values = {j1, j2};
        milvus::JSONFieldDataPtr cp = std::make_shared<milvus::JSONFieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), j2);
    }
}

TEST_F(FieldDataTest, VectorFields) {
    std::string name = "dummy";

    {
        milvus::FloatVecFieldData data{name};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::FLOAT_VECTOR);
        std::vector<float> element_1 = {1.0, 2.0};
        std::vector<float> element_2 = {3.0, 4.0};
        data.Add(element_1);
        data.Add(element_2);
        std::vector<float> element_3 = {5.0};
        EXPECT_EQ(data.Add(element_3), milvus::StatusCode::DIMENSION_NOT_EQUAL);
        std::vector<float> element_4;
        EXPECT_EQ(data.Add(element_4), milvus::StatusCode::VECTOR_IS_EMPTY);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0).size(), element_1.size());
        EXPECT_EQ(data.Data().at(1).size(), element_2.size());
        EXPECT_FLOAT_EQ(data.Data().at(0).at(0), element_1.at(0));
        EXPECT_FLOAT_EQ(data.Data().at(0).at(1), element_1.at(1));
        EXPECT_FLOAT_EQ(data.Data().at(1).at(0), element_2.at(0));
        EXPECT_FLOAT_EQ(data.Data().at(1).at(1), element_2.at(1));

        std::vector<std::vector<float>> values = {element_1, element_2};
        milvus::FloatVecFieldDataPtr cp = std::make_shared<milvus::FloatVecFieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), element_2);
    }

    {
        milvus::BinaryVecFieldData data{name};
        std::vector<uint8_t> element_1 = {1, 2};
        std::vector<uint8_t> element_2 = {3, 4};
        data.Add(element_1);
        data.Add(element_2);
        std::vector<uint8_t> element_3 = {5};
        EXPECT_EQ(data.Add(element_3), milvus::StatusCode::DIMENSION_NOT_EQUAL);
        std::vector<uint8_t> element_4;
        EXPECT_EQ(data.Add(element_4), milvus::StatusCode::VECTOR_IS_EMPTY);
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::BINARY_VECTOR);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0).size(), element_1.size());
        EXPECT_EQ(data.Data().at(1).size(), element_2.size());
        EXPECT_EQ(data.Data().at(0).at(0), element_1.at(0));
        EXPECT_EQ(data.Data().at(0).at(1), element_1.at(1));
        EXPECT_EQ(data.Data().at(1).at(0), element_2.at(0));
        EXPECT_EQ(data.Data().at(1).at(1), element_2.at(1));

        std::vector<std::vector<uint8_t>> values = {element_1, element_2};
        milvus::BinaryVecFieldDataPtr cp = std::make_shared<milvus::BinaryVecFieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), element_2);
    }

    {
        const std::vector<std::string> elements{"\x01\x02", "\x03\x04"};
        milvus::BinaryVecFieldData data{name, elements};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::BINARY_VECTOR);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.DataAsString().at(0), elements.at(0));
        EXPECT_EQ(data.DataAsString().at(1), elements.at(1));
        std::vector<std::vector<uint8_t>> expected{{1, 2}, {3, 4}};
        EXPECT_EQ(data.Data(), expected);

        const std::string element_1{'\x00', '\x00'};
        auto status = data.AddAsString(element_1);
        EXPECT_EQ(status, milvus::StatusCode::OK);
        expected = std::vector<std::vector<uint8_t>>{{1, 2}, {3, 4}, {0, 0}};
        EXPECT_EQ(data.Data(), expected);
    }

    {
        milvus::Float16VecFieldData data{name};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::FLOAT16_VECTOR);
        std::vector<float> origin_1 = {1.0, 2.0};
        std::vector<float> origin_2 = {3.0, 4.0};
        std::vector<uint16_t> element_1 = ToF16Vector(origin_1, false);
        std::vector<uint16_t> element_2 = ToF16Vector(origin_2, false);
        data.Add(element_1);
        data.Add(element_2);
        std::vector<float> element_3 = {5.0};
        EXPECT_EQ(data.Add(ToF16Vector(element_3, false)), milvus::StatusCode::DIMENSION_NOT_EQUAL);
        std::vector<float> element_4;
        EXPECT_EQ(data.Add(ToF16Vector(element_4, false)), milvus::StatusCode::VECTOR_IS_EMPTY);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0).size(), element_1.size());
        EXPECT_EQ(data.Data().at(1).size(), element_2.size());
        EXPECT_FLOAT_EQ(milvus::F16toF32(data.Data().at(0).at(0)), origin_1.at(0));
        EXPECT_FLOAT_EQ(milvus::F16toF32(data.Data().at(0).at(1)), origin_1.at(1));
        EXPECT_FLOAT_EQ(milvus::F16toF32(data.Data().at(1).at(0)), origin_2.at(0));
        EXPECT_FLOAT_EQ(milvus::F16toF32(data.Data().at(1).at(1)), origin_2.at(1));

        std::vector<std::vector<uint16_t>> values = {element_1, element_2};
        milvus::Float16VecFieldDataPtr cp = std::make_shared<milvus::Float16VecFieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), element_2);
    }

    {
        milvus::BFloat16VecFieldData data{name};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::BFLOAT16_VECTOR);
        std::vector<float> origin_1 = {1.0, 2.0};
        std::vector<float> origin_2 = {3.0, 4.0};
        std::vector<uint16_t> element_1 = ToF16Vector(origin_1, true);
        std::vector<uint16_t> element_2 = ToF16Vector(origin_2, true);
        data.Add(element_1);
        data.Add(element_2);
        std::vector<float> element_3 = {5.0};
        EXPECT_EQ(data.Add(ToF16Vector(element_3, true)), milvus::StatusCode::DIMENSION_NOT_EQUAL);
        std::vector<float> element_4;
        EXPECT_EQ(data.Add(ToF16Vector(element_4, true)), milvus::StatusCode::VECTOR_IS_EMPTY);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0).size(), element_1.size());
        EXPECT_EQ(data.Data().at(1).size(), element_2.size());
        EXPECT_FLOAT_EQ(milvus::BF16toF32(data.Data().at(0).at(0)), origin_1.at(0));
        EXPECT_FLOAT_EQ(milvus::BF16toF32(data.Data().at(0).at(1)), origin_1.at(1));
        EXPECT_FLOAT_EQ(milvus::BF16toF32(data.Data().at(1).at(0)), origin_2.at(0));
        EXPECT_FLOAT_EQ(milvus::BF16toF32(data.Data().at(1).at(1)), origin_2.at(1));

        std::vector<std::vector<uint16_t>> values = {element_1, element_2};
        milvus::BFloat16VecFieldDataPtr cp = std::make_shared<milvus::BFloat16VecFieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), element_2);
    }

    {
        milvus::SparseFloatVecFieldData data{name};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::SPARSE_FLOAT_VECTOR);
        std::map<uint32_t, float> element_1 = {{1, 0.4}, {5, 0.5}};
        std::map<uint32_t, float> element_2 = {{8, 0.1}, {100, 1.0}};
        data.Add(element_1);
        data.Add(element_2);
        EXPECT_EQ(data.Count(), 2);
        EXPECT_EQ(data.Data().size(), 2);
        EXPECT_EQ(data.Data().at(0), element_1);
        EXPECT_EQ(data.Data().at(1), element_2);

        std::vector<std::map<uint32_t, float>> values = {element_1, element_2};
        milvus::SparseFloatVecFieldDataPtr cp = std::make_shared<milvus::SparseFloatVecFieldData>(name, values);
        EXPECT_EQ(cp->Data().size(), 2);
        EXPECT_EQ(cp->Value(1), element_2);
    }
}

TEST_F(FieldDataTest, ArrayFields) {
    std::string name = "dummy";

    {
        const std::vector<milvus::ArrayBoolFieldData::ElementT> elements{{true, false}, {false}};
        milvus::ArrayBoolFieldData data{name, elements};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::ARRAY);
        EXPECT_EQ(data.ElementType(), milvus::DataType::BOOL);

        data.Add(elements.at(0));
        EXPECT_EQ(data.Count(), 3);
        EXPECT_EQ(data.Data().size(), 3);
        EXPECT_EQ(data.Data().at(0), elements.at(0));
        EXPECT_EQ(data.Data().at(1), elements.at(1));
        EXPECT_EQ(data.Data().at(2), elements.at(0));
    }

    {
        const std::vector<milvus::ArrayInt8FieldData::ElementT> elements{{2, 3}, {4}};
        milvus::ArrayInt8FieldData data{name, elements};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::ARRAY);
        EXPECT_EQ(data.ElementType(), milvus::DataType::INT8);

        data.Add(elements.at(0));
        EXPECT_EQ(data.Count(), 3);
        EXPECT_EQ(data.Data().size(), 3);
        EXPECT_EQ(data.Data().at(0), elements.at(0));
        EXPECT_EQ(data.Data().at(1), elements.at(1));
        EXPECT_EQ(data.Data().at(2), elements.at(0));
    }

    {
        const std::vector<milvus::ArrayInt16FieldData::ElementT> elements{{2, 3}, {4}};
        milvus::ArrayInt16FieldData data{name, elements};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::ARRAY);
        EXPECT_EQ(data.ElementType(), milvus::DataType::INT16);

        data.Add(elements.at(0));
        EXPECT_EQ(data.Count(), 3);
        EXPECT_EQ(data.Data().size(), 3);
        EXPECT_EQ(data.Data().at(0), elements.at(0));
        EXPECT_EQ(data.Data().at(1), elements.at(1));
        EXPECT_EQ(data.Data().at(2), elements.at(0));
    }

    {
        const std::vector<milvus::ArrayInt32FieldData::ElementT> elements{{2, 3}, {4}};
        milvus::ArrayInt32FieldData data{name, elements};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::ARRAY);
        EXPECT_EQ(data.ElementType(), milvus::DataType::INT32);

        data.Add(elements.at(0));
        EXPECT_EQ(data.Count(), 3);
        EXPECT_EQ(data.Data().size(), 3);
        EXPECT_EQ(data.Data().at(0), elements.at(0));
        EXPECT_EQ(data.Data().at(1), elements.at(1));
        EXPECT_EQ(data.Data().at(2), elements.at(0));
    }

    {
        const std::vector<milvus::ArrayInt64FieldData::ElementT> elements{{2, 3}, {4}};
        milvus::ArrayInt64FieldData data{name, elements};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::ARRAY);
        EXPECT_EQ(data.ElementType(), milvus::DataType::INT64);

        data.Add(elements.at(0));
        EXPECT_EQ(data.Count(), 3);
        EXPECT_EQ(data.Data().size(), 3);
        EXPECT_EQ(data.Data().at(0), elements.at(0));
        EXPECT_EQ(data.Data().at(1), elements.at(1));
        EXPECT_EQ(data.Data().at(2), elements.at(0));
    }

    {
        const std::vector<milvus::ArrayFloatFieldData::ElementT> elements{{0.2, 0.3}, {0.4}};
        milvus::ArrayFloatFieldData data{name, elements};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::ARRAY);
        EXPECT_EQ(data.ElementType(), milvus::DataType::FLOAT);

        data.Add(elements.at(0));
        EXPECT_EQ(data.Count(), 3);
        EXPECT_EQ(data.Data().size(), 3);
        EXPECT_EQ(data.Data().at(0), elements.at(0));
        EXPECT_EQ(data.Data().at(1), elements.at(1));
        EXPECT_EQ(data.Data().at(2), elements.at(0));
    }

    {
        const std::vector<milvus::ArrayDoubleFieldData::ElementT> elements{{0.2, 0.3}, {0.4}};
        milvus::ArrayDoubleFieldData data{name, elements};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::ARRAY);
        EXPECT_EQ(data.ElementType(), milvus::DataType::DOUBLE);

        data.Add(elements.at(0));
        EXPECT_EQ(data.Count(), 3);
        EXPECT_EQ(data.Data().size(), 3);
        EXPECT_EQ(data.Data().at(0), elements.at(0));
        EXPECT_EQ(data.Data().at(1), elements.at(1));
        EXPECT_EQ(data.Data().at(2), elements.at(0));
    }

    {
        const std::vector<milvus::ArrayVarCharFieldData::ElementT> elements{{"a", "bb"}, {"ccc"}};
        milvus::ArrayVarCharFieldData data{name, elements};
        EXPECT_EQ(data.Name(), name);
        EXPECT_EQ(data.Type(), milvus::DataType::ARRAY);
        EXPECT_EQ(data.ElementType(), milvus::DataType::VARCHAR);

        data.Add(elements.at(0));
        EXPECT_EQ(data.Count(), 3);
        EXPECT_EQ(data.Data().size(), 3);
        EXPECT_EQ(data.Data().at(0), elements.at(0));
        EXPECT_EQ(data.Data().at(1), elements.at(1));
        EXPECT_EQ(data.Data().at(2), elements.at(0));
    }
}