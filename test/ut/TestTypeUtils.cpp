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

TEST_F(TypeUtilsTest, BoolFieldNotEquals) {
    const std::string field_name = "foo";
    milvus::BoolFieldData bool_field{field_name, std::vector<bool>{false, true}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == bool_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_vectors();
    EXPECT_FALSE(proto_field == bool_field);

    auto scalars = proto_field.mutable_scalars();
    scalars->mutable_float_data();
    EXPECT_FALSE(proto_field == bool_field);

    auto bool_scalars = scalars->mutable_bool_data();
    bool_scalars->add_data(false);
    EXPECT_FALSE(proto_field == bool_field);
}

TEST_F(TypeUtilsTest, Int8FieldEqualsAndCast) {
    milvus::Int8FieldData int8_field_data{"foo", std::vector<int8_t>{1, 2}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(int8_field_data));
    auto int8_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, int8_field_data);
    EXPECT_EQ(proto_field_data, *int8_field_data_ptr);
    EXPECT_EQ(int8_field_data, *int8_field_data_ptr);
}

TEST_F(TypeUtilsTest, Int8FieldNotEquals) {
    const std::string field_name = "foo";
    milvus::Int8FieldData int8_field{field_name, std::vector<int8_t>{1, 2}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == int8_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_vectors();
    EXPECT_FALSE(proto_field == int8_field);

    auto scalars = proto_field.mutable_scalars();
    scalars->mutable_float_data();
    EXPECT_FALSE(proto_field == int8_field);

    auto int_scalars = scalars->mutable_int_data();
    int_scalars->add_data(1);
    EXPECT_FALSE(proto_field == int8_field);
}

TEST_F(TypeUtilsTest, Int16FieldEqualsAndCast) {
    milvus::Int16FieldData int16_field_data{"foo", std::vector<int16_t>{1, 2}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(int16_field_data));
    auto int16_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, int16_field_data);
    EXPECT_EQ(proto_field_data, *int16_field_data_ptr);
    EXPECT_EQ(int16_field_data, *int16_field_data_ptr);
}

TEST_F(TypeUtilsTest, Int16FieldNotEquals) {
    const std::string field_name = "foo";
    milvus::Int16FieldData int16_field{field_name, std::vector<int16_t>{1, 2}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == int16_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_vectors();
    EXPECT_FALSE(proto_field == int16_field);

    auto scalars = proto_field.mutable_scalars();
    scalars->mutable_float_data();
    EXPECT_FALSE(proto_field == int16_field);

    auto int_scalars = scalars->mutable_int_data();
    int_scalars->add_data(1);
    EXPECT_FALSE(proto_field == int16_field);
}

TEST_F(TypeUtilsTest, Int32FieldEqualsAndCast) {
    milvus::Int32FieldData int32_field_data{"foo", std::vector<int32_t>{1, 2}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(int32_field_data));
    auto int32_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, int32_field_data);
    EXPECT_EQ(proto_field_data, *int32_field_data_ptr);
    EXPECT_EQ(int32_field_data, *int32_field_data_ptr);
}

TEST_F(TypeUtilsTest, Int32FieldNotEquals) {
    const std::string field_name = "foo";
    milvus::Int32FieldData int32_field{field_name, std::vector<int32_t>{1, 2}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == int32_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_vectors();
    EXPECT_FALSE(proto_field == int32_field);

    auto scalars = proto_field.mutable_scalars();
    scalars->mutable_float_data();
    EXPECT_FALSE(proto_field == int32_field);

    auto int_scalars = scalars->mutable_int_data();
    int_scalars->add_data(1);
    EXPECT_FALSE(proto_field == int32_field);
}

TEST_F(TypeUtilsTest, Int64FieldEqualsAndCast) {
    milvus::Int64FieldData int64_field_data{"foo", std::vector<int64_t>{1, 2}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(int64_field_data));
    auto int64_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, int64_field_data);
    EXPECT_EQ(proto_field_data, *int64_field_data_ptr);
    EXPECT_EQ(int64_field_data, *int64_field_data_ptr);
}

TEST_F(TypeUtilsTest, Int64FieldNotEquals) {
    const std::string field_name = "foo";
    milvus::Int64FieldData int64_field{field_name, std::vector<int64_t>{1, 2}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == int64_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_vectors();
    EXPECT_FALSE(proto_field == int64_field);

    auto scalars = proto_field.mutable_scalars();
    scalars->mutable_float_data();
    EXPECT_FALSE(proto_field == int64_field);

    auto int_scalars = scalars->mutable_long_data();
    int_scalars->add_data(1);
    EXPECT_FALSE(proto_field == int64_field);
}

TEST_F(TypeUtilsTest, FloatFieldEqualsAndCast) {
    milvus::FloatFieldData float_field_data{"foo", std::vector<float>{0.1f, 0.2f}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(float_field_data));
    auto float_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, float_field_data);
    EXPECT_EQ(proto_field_data, *float_field_data_ptr);
    EXPECT_EQ(float_field_data, *float_field_data_ptr);
}

TEST_F(TypeUtilsTest, FloatFieldNotEquals) {
    const std::string field_name = "foo";
    milvus::FloatFieldData float_field{field_name, std::vector<float>{1.0f, 2.0f}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == float_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_vectors();
    EXPECT_FALSE(proto_field == float_field);

    auto scalars = proto_field.mutable_scalars();
    scalars->mutable_int_data();
    EXPECT_FALSE(proto_field == float_field);

    auto float_scalars = scalars->mutable_float_data();
    float_scalars->add_data(1.0);
    EXPECT_FALSE(proto_field == float_field);
}

TEST_F(TypeUtilsTest, DoubleFieldEqualsAndCast) {
    milvus::DoubleFieldData double_field_data{"foo", std::vector<double>{0.1, 0.2}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(double_field_data));
    auto double_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, double_field_data);
    EXPECT_EQ(proto_field_data, *double_field_data_ptr);
    EXPECT_EQ(double_field_data, *double_field_data_ptr);
}

TEST_F(TypeUtilsTest, DoubleFieldNotEquals) {
    const std::string field_name = "foo";
    milvus::DoubleFieldData double_field{field_name, std::vector<double>{1.0, 2.0}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == double_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_vectors();
    EXPECT_FALSE(proto_field == double_field);

    auto scalars = proto_field.mutable_scalars();
    scalars->mutable_int_data();
    EXPECT_FALSE(proto_field == double_field);

    auto double_scalars = scalars->mutable_double_data();
    double_scalars->add_data(1.0);
    EXPECT_FALSE(proto_field == double_field);
}

TEST_F(TypeUtilsTest, StringFieldEqualsAndCast) {
    milvus::StringFieldData string_field_data{"foo", std::vector<std::string>{"foo", "bar"}};
    auto proto_field_data = CreateProtoFieldData(static_cast<const milvus::Field&>(string_field_data));
    auto string_field_data_ptr = CreateMilvusFieldData(proto_field_data);
    EXPECT_EQ(proto_field_data, string_field_data);
    EXPECT_EQ(proto_field_data, *string_field_data_ptr);
    EXPECT_EQ(string_field_data, *string_field_data_ptr);
}

TEST_F(TypeUtilsTest, StringFieldNotEquals) {
    const std::string field_name = "foo";
    milvus::StringFieldData string_field{field_name, std::vector<std::string>{"a", "b"}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == string_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_vectors();
    EXPECT_FALSE(proto_field == string_field);

    auto scalars = proto_field.mutable_scalars();
    scalars->mutable_int_data();
    EXPECT_FALSE(proto_field == string_field);

    auto double_scalars = scalars->mutable_string_data();
    double_scalars->add_data("a");
    EXPECT_FALSE(proto_field == string_field);
}

TEST_F(TypeUtilsTest, BinaryVecFieldEqualsAndCast) {
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

TEST_F(TypeUtilsTest, BinaryVecFieldNotEquals) {
    const std::string field_name = "foo";
    milvus::BinaryVecFieldData bins_field{"foo", std::vector<std::vector<uint8_t>>{
                                                     std::vector<uint8_t>{1, 2},
                                                     std::vector<uint8_t>{3, 4},
                                                 }};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == bins_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_scalars();
    EXPECT_FALSE(proto_field == bins_field);

    auto scalars = proto_field.mutable_vectors();
    scalars->mutable_float_vector();
    EXPECT_FALSE(proto_field == bins_field);

    auto bins_scalars = scalars->mutable_binary_vector();
    bins_scalars->push_back('a');
    EXPECT_FALSE(proto_field == bins_field);

    bins_scalars->push_back('a');
    bins_scalars->push_back('a');
    bins_scalars->push_back('a');
    EXPECT_FALSE(proto_field == bins_field);
}

TEST_F(TypeUtilsTest, FloatVecFieldEqualsAndCast) {
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

TEST_F(TypeUtilsTest, FloatVecFieldNotEquals) {
    const std::string field_name = "foo";
    milvus::FloatVecFieldData floats_field{"foo", std::vector<std::vector<float>>{
                                                      std::vector<float>{0.1f, 0.2f},
                                                      std::vector<float>{0.3f, 0.4f},
                                                  }};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == floats_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_scalars();
    EXPECT_FALSE(proto_field == floats_field);

    auto scalars = proto_field.mutable_vectors();
    scalars->mutable_binary_vector();
    EXPECT_FALSE(proto_field == floats_field);

    auto floats_scalars = scalars->mutable_float_vector();
    floats_scalars->add_data(0.1f);
    EXPECT_FALSE(proto_field == floats_field);

    floats_scalars->add_data(0.1f);
    floats_scalars->add_data(0.1f);
    floats_scalars->add_data(0.1f);
    EXPECT_FALSE(proto_field == floats_field);
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

TEST_F(TypeUtilsTest, MetricTypeCastTest) {
    for (const auto& name :
         {"IP", "L2", "HAMMING", "SUBSTRUCTURE", "SUPERSTRUCTURE", "TANIMOTO", "JACCARD", "INVALID"}) {
        EXPECT_EQ(std::to_string(milvus::MetricTypeCast(name)), name);
    }
}

TEST_F(TypeUtilsTest, IndexTypeCastTest) {
    for (const auto& name : {"INVALID", "FLAT", "IVF_FLAT", "IVF_SQ8", "IVF_PQ", "HNSW", "IVF_HNSW", "RHNSW_FLAT",
                             "RHNSW_SQ", "RHNSW_PQ", "ANNOY", "BIN_FLAT", "BIN_IVF_FLAT"}) {
        EXPECT_EQ(std::to_string(milvus::IndexTypeCast(name)), name);
    }
}

TEST_F(TypeUtilsTest, DataTypeCast) {
    const std::vector<std::pair<milvus::DataType, milvus::proto::schema::DataType>> data_types = {
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

    for (auto& pair : data_types) {
        auto dt = milvus::DataTypeCast(milvus::DataType(pair.first));
        EXPECT_EQ(dt, pair.second);
    }
    for (auto& pair : data_types) {
        auto dt = milvus::DataTypeCast(milvus::proto::schema::DataType(pair.second));
        EXPECT_EQ(dt, pair.first);
    }
}

TEST_F(TypeUtilsTest, SegmentStateCast) {
    auto values = {milvus::SegmentState::DROPPED, milvus::SegmentState::FLUSHED,   milvus::SegmentState::FLUSHING,
                   milvus::SegmentState::GROWING, milvus::SegmentState::NOT_EXIST, milvus::SegmentState::SEALED,
                   milvus::SegmentState::UNKNOWN};
    for (auto value : values) {
        EXPECT_EQ(milvus::SegmentStateCast(milvus::SegmentStateCast(value)), value);
    }
}

TEST_F(TypeUtilsTest, IndexStateCast) {
    const std::vector<std::pair<int32_t, milvus::IndexStateCode>> states = {
        {milvus::proto::common::IndexState::IndexStateNone, milvus::IndexStateCode::NONE},
        {milvus::proto::common::IndexState::Unissued, milvus::IndexStateCode::UNISSUED},
        {milvus::proto::common::IndexState::InProgress, milvus::IndexStateCode::IN_PROGRESS},
        {milvus::proto::common::IndexState::Finished, milvus::IndexStateCode::FINISHED},
        {milvus::proto::common::IndexState::Failed, milvus::IndexStateCode::FAILED}};

    for (auto& pair : states) {
        auto it = milvus::IndexStateCast(milvus::proto::common::IndexState(pair.first));
        EXPECT_EQ(it, milvus::IndexStateCode(pair.second));
    }
}

TEST_F(TypeUtilsTest, ConvertFieldSchema) {
    const std::string field_name = "face";
    const std::string field_desc = "face signature";
    const bool primary_key = true;
    const bool auto_id = true;
    const milvus::DataType field_type = milvus::DataType::FLOAT_VECTOR;
    const uint32_t dimension = 128;
    milvus::FieldSchema field(field_name, field_type, field_desc, primary_key, auto_id);
    field.SetDimension(dimension);

    milvus::proto::schema::FieldSchema proto_field;
    milvus::ConvertFieldSchema(field, proto_field);

    EXPECT_EQ(proto_field.name(), field_name);
    EXPECT_EQ(proto_field.description(), field_desc);
    EXPECT_EQ(proto_field.is_primary_key(), primary_key);
    EXPECT_EQ(proto_field.autoid(), auto_id);
    EXPECT_EQ(proto_field.data_type(), milvus::DataTypeCast(field_type));

    milvus::FieldSchema sdk_field;
    milvus::ConvertFieldSchema(proto_field, sdk_field);
    EXPECT_EQ(sdk_field.Name(), field_name);
    EXPECT_EQ(sdk_field.Description(), field_desc);
    EXPECT_EQ(sdk_field.IsPrimaryKey(), primary_key);
    EXPECT_EQ(sdk_field.AutoID(), auto_id);
    EXPECT_EQ(sdk_field.FieldDataType(), field_type);
    EXPECT_EQ(sdk_field.Dimension(), dimension);
}

TEST_F(TypeUtilsTest, ConvertCollectionSchema) {
}
