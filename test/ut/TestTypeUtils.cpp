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

#include <gmock/gmock.h>

#include <functional>
#include <unordered_map>
#include <vector>

#include "TypeUtils.h"

using milvus::CreateIDArray;
using milvus::CreateMilvusFieldData;
using milvus::CreateProtoFieldData;
using ::testing::ElementsAre;

class TypeUtilsTest : public ::testing::Test {};

TEST_F(TypeUtilsTest, BoolFieldEqualsAndCast) {
    milvus::BoolFieldData bool_field_data{"foo", std::vector<bool>{false, true}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(bool_field_data));
    auto bool_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, bool_field_data);
    EXPECT_EQ(proto_field_data, *bool_field_data_ptr);
    EXPECT_EQ(bool_field_data, *bool_field_data_ptr);
}

TEST_F(TypeUtilsTest, Int8FieldEqualsAndCast) {
    milvus::Int8FieldData int8_field_data{"foo", std::vector<int8_t>{1, 2}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(int8_field_data));
    auto int8_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, int8_field_data);
    EXPECT_EQ(proto_field_data, *int8_field_data_ptr);
    EXPECT_EQ(int8_field_data, *int8_field_data_ptr);
}

TEST_F(TypeUtilsTest, Int16FieldEqualsAndCast) {
    milvus::Int16FieldData int16_field_data{"foo", std::vector<int16_t>{1, 2}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(int16_field_data));
    auto int16_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, int16_field_data);
    EXPECT_EQ(proto_field_data, *int16_field_data_ptr);
    EXPECT_EQ(int16_field_data, *int16_field_data_ptr);
}

TEST_F(TypeUtilsTest, Int32FieldEqualsAndCast) {
    milvus::Int32FieldData int32_field_data{"foo", std::vector<int32_t>{1, 2}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(int32_field_data));
    auto int32_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, int32_field_data);
    EXPECT_EQ(proto_field_data, *int32_field_data_ptr);
    EXPECT_EQ(int32_field_data, *int32_field_data_ptr);
}

TEST_F(TypeUtilsTest, Int64FieldEqualsAndCast) {
    milvus::Int64FieldData int64_field_data{"foo", std::vector<int64_t>{1, 2}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(int64_field_data));
    auto int64_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, int64_field_data);
    EXPECT_EQ(proto_field_data, *int64_field_data_ptr);
    EXPECT_EQ(int64_field_data, *int64_field_data_ptr);
}

TEST_F(TypeUtilsTest, FloatFieldEqualsAndCast) {
    milvus::FloatFieldData float_field_data{"foo", std::vector<float>{0.1f, 0.2f}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(float_field_data));
    auto float_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, float_field_data);
    EXPECT_EQ(proto_field_data, *float_field_data_ptr);
    EXPECT_EQ(float_field_data, *float_field_data_ptr);
}

TEST_F(TypeUtilsTest, DoubleFieldEqualsAndCast) {
    milvus::DoubleFieldData double_field_data{"foo", std::vector<double>{0.1, 0.2}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(double_field_data));
    auto double_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, double_field_data);
    EXPECT_EQ(proto_field_data, *double_field_data_ptr);
    EXPECT_EQ(double_field_data, *double_field_data_ptr);
}

TEST_F(TypeUtilsTest, StringFieldEqualsAndCast) {
    milvus::StringFieldData string_field_data{"foo", std::vector<std::string>{"foo", "bar"}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(string_field_data));
    auto string_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, string_field_data);
    EXPECT_EQ(proto_field_data, *string_field_data_ptr);
    EXPECT_EQ(string_field_data, *string_field_data_ptr);
}

TEST_F(TypeUtilsTest, BinsFieldEqualsAndCast) {
    milvus::BinaryVecFieldData bins_field_data{"foo", std::vector<std::vector<uint8_t>>{
                                                          std::vector<uint8_t>{1, 2},
                                                          std::vector<uint8_t>{3, 4},
                                                      }};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(bins_field_data));
    auto bins_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, bins_field_data);
    EXPECT_EQ(proto_field_data, *bins_field_data_ptr);
    EXPECT_EQ(bins_field_data, *bins_field_data_ptr);
}

TEST_F(TypeUtilsTest, FloatsFieldEqualsAndCast) {
    milvus::FloatVecFieldData floats_field_data{"foo", std::vector<std::vector<float>>{
                                                           std::vector<float>{0.1f, 0.2f},
                                                           std::vector<float>{0.3f, 0.4f},
                                                       }};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(floats_field_data));
    auto floats_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, floats_field_data);
    EXPECT_EQ(proto_field_data, *floats_field_data_ptr);
    EXPECT_EQ(floats_field_data, *floats_field_data_ptr);
}

TEST_F(TypeUtilsTest, IDArray) {
    milvus::proto::schema::IDs ids;
    ids.mutable_int_id()->add_data(10000);
    ids.mutable_int_id()->add_data(10001);
    auto id_array = CreateIDArray(ids);

    EXPECT_TRUE(id_array.IsIntegerID());
    EXPECT_THAT(id_array.IntIDArray(), ElementsAre(10000, 10001));

    ids.mutable_str_id()->add_data("10000");
    ids.mutable_str_id()->add_data("10001");
    id_array = CreateIDArray(ids);

    EXPECT_FALSE(id_array.IsIntegerID());
    EXPECT_THAT(id_array.StrIDArray(), ElementsAre("10000", "10001"));
}

TEST_F(TypeUtilsTest, IDArrayWithRange) {
    milvus::proto::schema::IDs ids;
    ids.mutable_int_id()->add_data(10000);
    ids.mutable_int_id()->add_data(10001);
    ids.mutable_int_id()->add_data(10002);
    ids.mutable_int_id()->add_data(10003);
    auto id_array = CreateIDArray(ids, 1, 2);

    EXPECT_TRUE(id_array.IsIntegerID());
    EXPECT_THAT(id_array.IntIDArray(), ElementsAre(10001, 10002));

    ids.mutable_str_id()->add_data("10000");
    ids.mutable_str_id()->add_data("10001");
    ids.mutable_str_id()->add_data("10002");
    ids.mutable_str_id()->add_data("10003");
    id_array = CreateIDArray(ids, 1, 2);

    EXPECT_FALSE(id_array.IsIntegerID());
    EXPECT_THAT(id_array.StrIDArray(), ElementsAre("10001", "10002"));
}

TEST_F(TypeUtilsTest, CreateMilvusFieldDataWithRange) {
    milvus::BoolFieldData bool_field_data{"foo", std::vector<bool>{false, true, false}};
    const auto bool_field_data_ptr = std::dynamic_pointer_cast<const milvus::BoolFieldData>(
        CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(bool_field_data)), 1, 2));
    EXPECT_THAT(bool_field_data_ptr->Data(), ElementsAre(true, false));

    milvus::Int8FieldData int8_field_data{"foo", std::vector<int8_t>{1, 2, 1}};
    const auto int8_field_data_ptr = std::dynamic_pointer_cast<const milvus::Int8FieldData>(
        CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(int8_field_data)), 1, 2));
    EXPECT_THAT(int8_field_data_ptr->Data(), ElementsAre(2, 1));

    milvus::Int16FieldData int16_field_data{"foo", std::vector<int16_t>{1, 2, 1}};
    const auto int16_field_data_ptr = std::dynamic_pointer_cast<const milvus::Int16FieldData>(
        CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(int16_field_data)), 1, 2));
    EXPECT_THAT(int16_field_data_ptr->Data(), ElementsAre(2, 1));

    milvus::Int32FieldData int32_field_data{"foo", std::vector<int32_t>{1, 2, 1}};
    const auto int32_field_data_ptr = std::dynamic_pointer_cast<const milvus::Int32FieldData>(
        CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(int32_field_data)), 1, 2));
    EXPECT_THAT(int32_field_data_ptr->Data(), ElementsAre(2, 1));

    milvus::Int64FieldData int64_field_data{"foo", std::vector<int64_t>{1, 2, 1}};
    const auto int64_field_data_ptr = std::dynamic_pointer_cast<const milvus::Int64FieldData>(
        CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(int64_field_data)), 1, 2));
    EXPECT_THAT(int64_field_data_ptr->Data(), ElementsAre(2, 1));

    milvus::FloatFieldData float_field_data{"foo", std::vector<float>{0.1f, 0.2f, 0.3f}};
    const auto float_field_data_ptr = std::dynamic_pointer_cast<const milvus::FloatFieldData>(
        CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(float_field_data)), 1, 2));
    EXPECT_THAT(float_field_data_ptr->Data(), ElementsAre(0.2f, 0.3f));

    milvus::DoubleFieldData double_field_data{"foo", std::vector<double>{0.1, 0.2, 0.3}};
    const auto double_field_data_ptr = std::dynamic_pointer_cast<const milvus::DoubleFieldData>(
        CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(double_field_data)), 1, 2));
    EXPECT_THAT(double_field_data_ptr->Data(), ElementsAre(0.2, 0.3));

    milvus::StringFieldData string_field_data{"foo", std::vector<std::string>{"a", "b", "c"}};
    const auto string_field_data_ptr = std::dynamic_pointer_cast<const milvus::StringFieldData>(
        CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(string_field_data)), 1, 2));
    EXPECT_THAT(string_field_data_ptr->Data(), ElementsAre("b", "c"));

    milvus::BinaryVecFieldData bins_field_data{"foo",
                                               std::vector<std::vector<uint8_t>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
    const auto bins_field_data_ptr = std::dynamic_pointer_cast<const milvus::BinaryVecFieldData>(
        CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(bins_field_data)), 1, 2));
    EXPECT_THAT(bins_field_data_ptr->Data(), ElementsAre(std::vector<uint8_t>{4, 5, 6}, std::vector<uint8_t>{7, 8, 9}));

    milvus::FloatVecFieldData floats_field_data{
        "foo", std::vector<std::vector<float>>{{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}, {0.7f, 0.8f, 0.9f}}};
    const auto floats_field_data_ptr = std::dynamic_pointer_cast<const milvus::FloatVecFieldData>(
        CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(floats_field_data)), 1, 2));
    EXPECT_THAT(floats_field_data_ptr->Data(),
                ElementsAre(std::vector<float>{0.4f, 0.5f, 0.6f}, std::vector<float>{0.7f, 0.8f, 0.9f}));
}

TEST_F(TypeUtilsTest, MetricTypeToString) {
    EXPECT_EQ(std::to_string(milvus::MetricType::IP), "IP");
    EXPECT_EQ(std::to_string(milvus::MetricType::L2), "L2");
    EXPECT_EQ(std::to_string(milvus::MetricType::HAMMING), "HAMMING");
    EXPECT_EQ(std::to_string(milvus::MetricType::SUBSTRUCTURE), "SUBSTRUCTURE");
    EXPECT_EQ(std::to_string(milvus::MetricType::SUPERSTRUCTURE), "SUPERSTRUCTURE");
    EXPECT_EQ(std::to_string(milvus::MetricType::TANIMOTO), "TANIMOTO");
    EXPECT_EQ(std::to_string(milvus::MetricType::JACCARD), "JACCARD");
    EXPECT_EQ(std::to_string(milvus::MetricType::INVALID), "INVALID");
}

TEST_F(TypeUtilsTest, DataTypeCast) {
    const std::unordered_map<int32_t, int32_t> sdk2proto = {
        {milvus::DataType::UNKNOWN, milvus::proto::schema::DataType::None},
        {milvus::DataType::BOOL, milvus::proto::schema::DataType::Bool},
        {milvus::DataType::INT8, milvus::proto::schema::DataType::Int8},
        {milvus::DataType::INT16, milvus::proto::schema::DataType::Int16},
        {milvus::DataType::INT32, milvus::proto::schema::DataType::Int32},
        {milvus::DataType::INT64, milvus::proto::schema::DataType::Int64},
        {milvus::DataType::FLOAT, milvus::proto::schema::DataType::Float},
        {milvus::DataType::DOUBLE, milvus::proto::schema::DataType::Double},
        {milvus::DataType::STRING, milvus::proto::schema::DataType::String},
        {milvus::DataType::FLOAT_VECTOR, milvus::proto::schema::DataType::FloatVector},
        {milvus::DataType::BINARY_VECTOR, milvus::proto::schema::DataType::BinaryVector}};

    for (auto& pair : sdk2proto) {
        auto dt = milvus::DataTypeCast(milvus::DataType(pair.first));
        EXPECT_EQ(dt, pair.second);
    }

    std::unordered_map<int32_t, int32_t> proto2sdk;
    for (auto& pair : sdk2proto) {
        proto2sdk.insert(std::make_pair(pair.second, pair.first));
    }

    for (auto& pair : proto2sdk) {
        auto dt = milvus::DataTypeCast(milvus::proto::schema::DataType(pair.first));
        EXPECT_EQ(dt, pair.second);
    }
}