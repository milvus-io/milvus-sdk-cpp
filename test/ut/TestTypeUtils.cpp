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

#include <vector>

#include "utils/CompareUtils.h"
#include "utils/DmlUtils.h"
#include "utils/DqlUtils.h"
#include "utils/TypeUtils.h"

using ::testing::ElementsAre;

class TypeUtilsTest : public ::testing::Test {};

TEST_F(TypeUtilsTest, BoolFieldCompare) {
    const std::string field_name = "foo";
    milvus::BoolFieldData bool_field{field_name, std::vector<bool>{false, true}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_type(milvus::proto::schema::DataType::Bool);
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

    bool_scalars->add_data(true);
    EXPECT_TRUE(proto_field == bool_field);
}

TEST_F(TypeUtilsTest, Int8FieldCompare) {
    const std::string field_name = "foo";
    milvus::Int8FieldData int8_field{field_name, std::vector<int8_t>{1, 2}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_type(milvus::proto::schema::DataType::Int8);
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

    int_scalars->add_data(2);
    EXPECT_TRUE(proto_field == int8_field);
}

TEST_F(TypeUtilsTest, Int16FieldCompare) {
    const std::string field_name = "foo";
    milvus::Int16FieldData int16_field{field_name, std::vector<int16_t>{1, 2}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_type(milvus::proto::schema::DataType::Int16);
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

    int_scalars->add_data(2);
    EXPECT_TRUE(proto_field == int16_field);
}

TEST_F(TypeUtilsTest, Int32FieldCompare) {
    const std::string field_name = "foo";
    milvus::Int32FieldData int32_field{field_name, std::vector<int32_t>{1, 2}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_type(milvus::proto::schema::DataType::Int32);
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

    int_scalars->add_data(2);
    EXPECT_TRUE(proto_field == int32_field);
}

TEST_F(TypeUtilsTest, Int64FieldCompare) {
    const std::string field_name = "foo";
    milvus::Int64FieldData int64_field{field_name, std::vector<int64_t>{1, 2}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_type(milvus::proto::schema::DataType::Int64);
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

    int_scalars->add_data(2);
    EXPECT_TRUE(proto_field == int64_field);
}

TEST_F(TypeUtilsTest, FloatFieldCompare) {
    const std::string field_name = "foo";
    milvus::FloatFieldData float_field{field_name, std::vector<float>{1.0f, 2.0f}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_type(milvus::proto::schema::DataType::Float);
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

    float_scalars->add_data(2.0);
    EXPECT_TRUE(proto_field == float_field);
}

TEST_F(TypeUtilsTest, DoubleFieldCompare) {
    const std::string field_name = "foo";
    milvus::DoubleFieldData double_field{field_name, std::vector<double>{1.0, 2.0}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_type(milvus::proto::schema::DataType::Double);
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

    double_scalars->add_data(2.0);
    EXPECT_TRUE(proto_field == double_field);
}

TEST_F(TypeUtilsTest, StringFieldCompare) {
    const std::string field_name = "foo";
    milvus::VarCharFieldData string_field{field_name, std::vector<std::string>{"a", "b"}};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_type(milvus::proto::schema::DataType::VarChar);
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == string_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_vectors();
    EXPECT_FALSE(proto_field == string_field);

    auto scalars = proto_field.mutable_scalars();
    scalars->mutable_int_data();
    EXPECT_FALSE(proto_field == string_field);

    auto str_scalars = scalars->mutable_string_data();
    str_scalars->add_data("a");
    EXPECT_FALSE(proto_field == string_field);

    str_scalars->add_data("b");
    EXPECT_TRUE(proto_field == string_field);
}

TEST_F(TypeUtilsTest, JSONFieldCompare) {
    const std::string field_name = "foo";
    auto values = std::vector<nlohmann::json>{R"({"name":"aaa","age":18,"score":88})"};
    milvus::JSONFieldData json_field{field_name, values};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_type(milvus::proto::schema::DataType::JSON);
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == json_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_vectors();
    EXPECT_FALSE(proto_field == json_field);

    auto scalars = proto_field.mutable_scalars();
    scalars->mutable_int_data();
    EXPECT_FALSE(proto_field == json_field);

    auto json_scalars = scalars->mutable_json_data();
    json_scalars->add_data(values.at(0).dump());
    EXPECT_TRUE(proto_field == json_field);

    nlohmann::json a1 = R"({"name":"aaa","age":18,"score":88})";
    nlohmann::json a2 = R"({"name":"aaa","age":17,"score":77})";
    json_field.Add(a1);
    json_scalars->add_data(a2.dump());
    EXPECT_FALSE(proto_field == json_field);
}

TEST_F(TypeUtilsTest, BinaryVecFieldCompare) {
    const std::string field_name = "foo";
    milvus::BinaryVecFieldData bins_field{"foo", std::vector<std::vector<uint8_t>>{
                                                     std::vector<uint8_t>{1, 2},
                                                     std::vector<uint8_t>{3, 4},
                                                 }};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_type(milvus::proto::schema::DataType::BinaryVector);
    proto_field.set_field_name("_");
    EXPECT_FALSE(proto_field == bins_field);

    proto_field.set_field_name(field_name);
    proto_field.mutable_scalars();
    EXPECT_FALSE(proto_field == bins_field);

    auto scalars = proto_field.mutable_vectors();
    scalars->mutable_float_vector();
    EXPECT_FALSE(proto_field == bins_field);

    auto bins_scalars = scalars->mutable_binary_vector();
    bins_scalars->push_back(static_cast<char>(1));
    bins_scalars->push_back(static_cast<char>(2));
    EXPECT_FALSE(proto_field == bins_field);

    bins_scalars->push_back(static_cast<char>(3));
    bins_scalars->push_back(static_cast<char>(4));
    EXPECT_TRUE(proto_field == bins_field);
}

TEST_F(TypeUtilsTest, FloatVecFieldCompare) {
    const std::string field_name = "foo";
    milvus::FloatVecFieldData floats_field{"foo", std::vector<std::vector<float>>{
                                                      std::vector<float>{0.1f, 0.2f},
                                                      std::vector<float>{0.3f, 0.4f},
                                                  }};
    milvus::proto::schema::FieldData proto_field;
    proto_field.set_type(milvus::proto::schema::DataType::FloatVector);
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
    floats_scalars->add_data(0.2f);
    EXPECT_FALSE(proto_field == floats_field);

    floats_scalars->add_data(0.3f);
    floats_scalars->add_data(0.4f);
    EXPECT_TRUE(proto_field == floats_field);
}

TEST_F(TypeUtilsTest, MetricTypeCastTest) {
    for (const auto& name : {"IP", "L2", "COSINE", "HAMMING", "JACCARD", "BM25", "DEFAULT"}) {
        EXPECT_EQ(std::to_string(milvus::MetricTypeCast(name)), name);
    }
}

TEST_F(TypeUtilsTest, IndexTypeCastTest) {
    for (const auto& name : {"INVALID",    "FLAT",         "IVF_FLAT",     "IVF_SQ8",
                             "IVF_PQ",     "HNSW",         "DISKANN",      "AUTOINDEX",
                             "SCANN",      "GPU_IVF_FLAT", "GPU_IVF_PQ",   "GPU_BRUTE_FORCE",
                             "GPU_CAGRA",  "BIN_FLAT",     "BIN_IVF_FLAT", "Trie",
                             "STL_SORT",   "INVERTED",     "BITMAP",       "SPARSE_INVERTED_INDEX",
                             "SPARSE_WAND"}) {
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
        {milvus::DataType::VARCHAR, milvus::proto::schema::DataType::VarChar},
        {milvus::DataType::JSON, milvus::proto::schema::DataType::JSON},
        {milvus::DataType::ARRAY, milvus::proto::schema::DataType::Array},
        {milvus::DataType::FLOAT_VECTOR, milvus::proto::schema::DataType::FloatVector},
        {milvus::DataType::BINARY_VECTOR, milvus::proto::schema::DataType::BinaryVector},
        {milvus::DataType::FLOAT16_VECTOR, milvus::proto::schema::DataType::Float16Vector},
        {milvus::DataType::BFLOAT16_VECTOR, milvus::proto::schema::DataType::BFloat16Vector},
        {milvus::DataType::SPARSE_FLOAT_VECTOR, milvus::proto::schema::DataType::SparseFloatVector}};

    for (auto& pair : data_types) {
        auto dt = milvus::DataTypeCast(milvus::DataType(pair.first));
        EXPECT_EQ(dt, pair.second);
    }
    for (auto& pair : data_types) {
        auto dt = milvus::DataTypeCast(milvus::proto::schema::DataType(pair.second));
        EXPECT_EQ(dt, pair.first);
    }
}

TEST_F(TypeUtilsTest, FunctionTypeCastTest) {
    auto values = {milvus::FunctionType::UNKNOWN, milvus::FunctionType::BM25};
    for (auto value : values) {
        EXPECT_EQ(milvus::FunctionTypeCast(milvus::FunctionTypeCast(value)), value);
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

TEST_F(TypeUtilsTest, ConvertValueFieldSchema) {
    const std::vector<std::pair<milvus::DataType, nlohmann::json>> valid_pairs = {
        {milvus::DataType::UNKNOWN, nlohmann::json()},  // null json, directly return ok
        {milvus::DataType::BOOL, nlohmann::json(true)},
        {milvus::DataType::INT8, nlohmann::json(6)},
        {milvus::DataType::INT16, nlohmann::json(60)},
        {milvus::DataType::INT32, nlohmann::json(600)},
        {milvus::DataType::INT64, nlohmann::json(-6000)},
        {milvus::DataType::FLOAT, nlohmann::json(3.14)},
        {milvus::DataType::FLOAT, nlohmann::json(3)},
        {milvus::DataType::DOUBLE, nlohmann::json(9.99)},
        {milvus::DataType::DOUBLE, nlohmann::json(9)},
        {milvus::DataType::VARCHAR, nlohmann::json("ok")},
        {milvus::DataType::JSON, nlohmann::json(R"([1, 2, 3, 4])")},
    };
    for (auto& pair : valid_pairs) {
        const milvus::FieldSchema field =
            milvus::FieldSchema().WithName("dummy").WithDataType(pair.first).WithDefaultValue(pair.second);
        auto status = milvus::CheckDefaultValue(field);
        EXPECT_TRUE(status.IsOk());

        milvus::proto::schema::FieldSchema proto_field;
        ConvertValueFieldSchema(pair.second, pair.first, *proto_field.mutable_default_value());

        nlohmann::json converted_json;
        ConvertValueFieldSchema(proto_field.default_value(), pair.first, converted_json);
        if (converted_json.is_number()) {
            // for numeric value, format to double to compare with tolerance
            milvus::IsNumEquals(converted_json.get<double>(), pair.second.get<double>());
        } else {
            EXPECT_EQ(converted_json, pair.second);
        }
    }

    const std::vector<std::pair<milvus::DataType, nlohmann::json>> invalid_pairs = {
        {milvus::DataType::BOOL, nlohmann::json(5)},
        {milvus::DataType::INT8, nlohmann::json("ok")},
        {milvus::DataType::INT16, nlohmann::json("ok")},
        {milvus::DataType::INT32, nlohmann::json("ok")},
        {milvus::DataType::INT64, nlohmann::json("ok")},
        {milvus::DataType::INT8, nlohmann::json(3.1)},
        {milvus::DataType::INT16, nlohmann::json(3.2)},
        {milvus::DataType::INT32, nlohmann::json(3.3)},
        {milvus::DataType::INT64, nlohmann::json(3.4)},
        {milvus::DataType::FLOAT, nlohmann::json("ok")},
        {milvus::DataType::DOUBLE, nlohmann::json("ok")},
        {milvus::DataType::VARCHAR, nlohmann::json(1)},
        {milvus::DataType::VARCHAR, nlohmann::json(false)},
        {milvus::DataType::BINARY_VECTOR, nlohmann::json(1)},
        {milvus::DataType::FLOAT_VECTOR, nlohmann::json(1)},
        {milvus::DataType::FLOAT16_VECTOR, nlohmann::json(1)},
        {milvus::DataType::BFLOAT16_VECTOR, nlohmann::json(1)},
        {milvus::DataType::SPARSE_FLOAT_VECTOR, nlohmann::json(1)},
    };
    for (auto& pair : invalid_pairs) {
        const milvus::FieldSchema field =
            milvus::FieldSchema().WithName("dummy").WithDataType(pair.first).WithDefaultValue(pair.second);
        auto status = milvus::CheckDefaultValue(field);
        EXPECT_FALSE(status.IsOk());
    }
}

TEST_F(TypeUtilsTest, ConvertFieldSchema) {
    const std::string field_name = "face";
    const std::string field_desc = "face signature";
    const bool primary_key = true;
    const bool auto_id = true;
    milvus::DataType field_type = milvus::DataType::FLOAT_VECTOR;
    const milvus::DataType element_type = milvus::DataType::DOUBLE;
    const uint32_t dimension = 128;

    {
        milvus::FieldSchema field = milvus::FieldSchema()
                                        .WithName(field_name)
                                        .WithDescription(field_desc)
                                        .WithDataType(field_type)
                                        .WithPrimaryKey(primary_key)
                                        .WithAutoID(auto_id)
                                        .WithElementType(element_type)
                                        .WithDimension(dimension)
                                        .WithMaxLength(100)
                                        .WithMaxCapacity(200)
                                        .WithPartitionKey(true)
                                        .WithClusteringKey(true)
                                        .EnableMatch(true)
                                        .EnableAnalyzer(true)
                                        .WithNullable(true)
                                        .WithDefaultValue("aaa");
        EXPECT_EQ(field.Name(), field_name);
        EXPECT_EQ(field.Description(), field_desc);
        EXPECT_EQ(field.FieldDataType(), field_type);
        EXPECT_EQ(field.IsPrimaryKey(), primary_key);
        EXPECT_EQ(field.AutoID(), auto_id);
        EXPECT_EQ(field.ElementType(), element_type);
        EXPECT_EQ(field.Dimension(), dimension);
        EXPECT_EQ(field.MaxLength(), 100);
        EXPECT_EQ(field.MaxCapacity(), 200);
        EXPECT_EQ(field.IsPartitionKey(), true);
        EXPECT_EQ(field.IsClusteringKey(), true);
        EXPECT_EQ(field.IsEnableAnalyzer(), true);
        EXPECT_EQ(field.IsEnableMatch(), true);
        EXPECT_EQ(field.IsNullable(), true);
        EXPECT_EQ(field.DefaultValue(), "aaa");
    }

    {
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
        EXPECT_EQ(sdk_field.IsEnableAnalyzer(), false);
        EXPECT_EQ(sdk_field.IsEnableMatch(), false);
        EXPECT_EQ(sdk_field.IsPartitionKey(), false);
        EXPECT_EQ(sdk_field.IsClusteringKey(), false);
    }

    field_type = milvus::DataType::ARRAY;
    const uint32_t capacity = 128;
    {
        milvus::FieldSchema field(field_name, field_type, field_desc, primary_key, auto_id);
        field.SetElementType(element_type);
        field.SetMaxCapacity(capacity);

        milvus::proto::schema::FieldSchema proto_field;
        milvus::ConvertFieldSchema(field, proto_field);

        EXPECT_EQ(proto_field.data_type(), milvus::DataTypeCast(field_type));
        EXPECT_EQ(proto_field.element_type(), milvus::DataTypeCast(element_type));

        milvus::FieldSchema sdk_field;
        milvus::ConvertFieldSchema(proto_field, sdk_field);
        EXPECT_EQ(sdk_field.FieldDataType(), field_type);
        EXPECT_EQ(sdk_field.ElementType(), element_type);
        EXPECT_EQ(sdk_field.MaxCapacity(), capacity);
    }

    field_type = milvus::DataType::VARCHAR;
    const uint32_t length = 512;
    nlohmann::json analyzer_params = {
        {"tokenizer", "standard"},
        {"filter", {"lowercase", {{"type", "length"}, {"max", 40}}, {{"type", "stop"}, {"stop_words", {"of", "for"}}}}},
    };
    {
        milvus::FieldSchema field(field_name, field_type, field_desc, primary_key, auto_id);
        field.SetMaxLength(length);
        field.EnableAnalyzer(true);
        field.EnableMatch(true);
        field.SetAnalyzerParams(analyzer_params);
        field.SetPartitionKey(true);
        field.SetClusteringKey(true);
        field.SetNullable(false);
        field.SetDefaultValue("abc");

        milvus::proto::schema::FieldSchema proto_field;
        milvus::ConvertFieldSchema(field, proto_field);

        EXPECT_EQ(proto_field.data_type(), milvus::DataTypeCast(field_type));

        milvus::FieldSchema sdk_field;
        milvus::ConvertFieldSchema(proto_field, sdk_field);
        EXPECT_EQ(sdk_field.FieldDataType(), field_type);
        EXPECT_EQ(sdk_field.MaxLength(), length);
        EXPECT_EQ(sdk_field.IsEnableAnalyzer(), true);
        EXPECT_EQ(sdk_field.IsEnableMatch(), true);
        EXPECT_EQ(sdk_field.AnalyzerParams(), analyzer_params);
        EXPECT_EQ(sdk_field.IsPartitionKey(), true);
        EXPECT_EQ(sdk_field.IsClusteringKey(), true);
        EXPECT_EQ(sdk_field.IsNullable(), false);
        EXPECT_EQ(sdk_field.DefaultValue(), "abc");
    }
}

TEST_F(TypeUtilsTest, ConvertCollectionSchema) {
    const std::string collection_name = "dummy";
    const std::string collection_desc = "desc";
    const bool enable_dynamic = true;
    const std::string field_name = "field";
    const std::string function_name = "bm25_func";
    const std::string function_desc = "bm25 function";
    const milvus::FunctionType function_type = milvus::FunctionType::BM25;

    milvus::CollectionSchema collection_schema(collection_name, collection_desc, 1, enable_dynamic);
    collection_schema.AddField({field_name, milvus::DataType::INT64, "dummy", true, true});

    milvus::FunctionPtr function = std::make_shared<milvus::Function>(function_name, function_type, function_desc);
    function->AddInputFieldName("aaa");
    function->AddOutputFieldName("bbb");
    function->AddParam("111", "222");
    collection_schema.AddFunction(function);

    milvus::proto::schema::CollectionSchema proto_schema;
    ConvertCollectionSchema(collection_schema, proto_schema);

    EXPECT_EQ(proto_schema.name(), collection_name);
    EXPECT_EQ(proto_schema.description(), collection_desc);
    EXPECT_EQ(proto_schema.enable_dynamic_field(), enable_dynamic);
    EXPECT_EQ(proto_schema.fields_size(), 1);
    EXPECT_EQ(proto_schema.functions_size(), 1);

    milvus::CollectionSchema sdk_schema;
    ConvertCollectionSchema(proto_schema, sdk_schema);

    EXPECT_EQ(sdk_schema.Name(), collection_name);
    EXPECT_EQ(sdk_schema.Description(), collection_desc);
    EXPECT_EQ(sdk_schema.EnableDynamicField(), enable_dynamic);
    EXPECT_EQ(sdk_schema.Fields().size(), 1);
    EXPECT_EQ(sdk_schema.Fields().at(0).Name(), field_name);
    EXPECT_EQ(sdk_schema.Functions().size(), 1);
    milvus::FunctionPtr sdk_function = sdk_schema.Functions().at(0);
    EXPECT_EQ(sdk_function->Name(), function_name);
    EXPECT_EQ(sdk_function->Description(), function_desc);
    EXPECT_EQ(sdk_function->GetFunctionType(), function_type);
    EXPECT_EQ(sdk_function->InputFieldNames().size(), 1);
    EXPECT_EQ(sdk_function->InputFieldNames().at(0), "aaa");
    EXPECT_EQ(sdk_function->OutputFieldNames().size(), 1);
    EXPECT_EQ(sdk_function->OutputFieldNames().at(0), "bbb");
    EXPECT_EQ(sdk_function->Params().size(), 1);
    EXPECT_EQ(sdk_function->Params().count("111"), 1);
    EXPECT_EQ(sdk_function->Params().at("111"), "222");
}

TEST_F(TypeUtilsTest, TestB64EncodeGeneric) {
    EXPECT_EQ(milvus::Base64Encode(""), "");
    EXPECT_EQ(milvus::Base64Encode("a"), "YQ==");
    EXPECT_EQ(milvus::Base64Encode("ab"), "YWI=");
    EXPECT_EQ(milvus::Base64Encode("abc"), "YWJj");
    EXPECT_EQ(milvus::Base64Encode("abcd"), "YWJjZA==");
    EXPECT_EQ(milvus::Base64Encode("abcde"), "YWJjZGU=");
}

TEST_F(TypeUtilsTest, ConsistencyLevelCast) {
    auto proto_levels = std::vector<milvus::proto::common::ConsistencyLevel>{
        milvus::proto::common::ConsistencyLevel::Strong,
        milvus::proto::common::ConsistencyLevel::Session,
        milvus::proto::common::ConsistencyLevel::Bounded,
        milvus::proto::common::ConsistencyLevel::Eventually,
    };
    auto sdk_levels = std::vector<milvus::ConsistencyLevel>{
        milvus::ConsistencyLevel::STRONG,
        milvus::ConsistencyLevel::SESSION,
        milvus::ConsistencyLevel::BOUNDED,
        milvus::ConsistencyLevel::EVENTUALLY,
    };

    for (size_t i = 0; i < proto_levels.size(); ++i) {
        auto sdk_level = milvus::ConsistencyLevelCast(proto_levels[i]);
        EXPECT_EQ(sdk_levels[i], sdk_level);
    }
    for (size_t i = 0; i < sdk_levels.size(); ++i) {
        auto proto_level = milvus::ConsistencyLevelCast(sdk_levels[i]);
        EXPECT_EQ(proto_levels[i], proto_level);
    }
}
