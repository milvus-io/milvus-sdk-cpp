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

#include "milvus/types/Constants.h"
#include "milvus/types/FieldData.h"
#include "milvus/types/FieldSchema.h"
#include "utils/Constants.h"
#include "utils/DmlUtils.h"
#include "utils/DqlUtils.h"

using milvus::CreateIDArray;
using milvus::CreateMilvusFieldData;
using milvus::CreateProtoFieldData;
using ::testing::ElementsAre;

class DmlUtilsTest : public ::testing::Test {};

TEST_F(DmlUtilsTest, IDArray) {
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

TEST_F(DmlUtilsTest, CreateMilvusFieldDataWithRange_Scalar) {
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

    milvus::VarCharFieldData string_field_data{"foo", std::vector<std::string>{"a", "b", "c"}};
    const auto string_field_data_ptr = std::dynamic_pointer_cast<const milvus::VarCharFieldData>(
        CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(string_field_data)), 1, 2));
    EXPECT_THAT(string_field_data_ptr->Data(), ElementsAre("b", "c"));

    auto values =
        std::vector<nlohmann::json>{R"({"name":"aaa","age":18,"score":88})", R"({"name":"bbb","age":19,"score":99})",
                                    R"({"name":"ccc","age":15,"score":100})"};
    milvus::JSONFieldData json_field_data{"foo", values};
    const auto json_field_data_ptr = std::dynamic_pointer_cast<const milvus::JSONFieldData>(
        CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(json_field_data)), 1, 2));
    EXPECT_THAT(json_field_data_ptr->Data(), ElementsAre(values.at(1), values.at(2)));
}

TEST_F(DmlUtilsTest, CreateMilvusFieldDataWithRange_Vector) {
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

TEST_F(DmlUtilsTest, CreateMilvusFieldDataWithRange_Array) {
    const std::string name = "foo";
    {
        auto values = std::vector<milvus::ArrayBoolFieldData::ElementT>{{true, false}, {false}};
        milvus::ArrayBoolFieldData field_data{name, values};
        auto field_data_ptr = std::dynamic_pointer_cast<const milvus::ArrayBoolFieldData>(
            CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(field_data)), 10, 20));
        EXPECT_TRUE(field_data_ptr->Data().empty());

        field_data_ptr = std::dynamic_pointer_cast<const milvus::ArrayBoolFieldData>(
            CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(field_data)), 10, 10));
        EXPECT_TRUE(field_data_ptr->Data().empty());

        field_data_ptr = std::dynamic_pointer_cast<const milvus::ArrayBoolFieldData>(
            CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(field_data)), -5, 1));
        EXPECT_THAT(field_data_ptr->Data(), ElementsAre(values.at(0)));

        field_data_ptr = std::dynamic_pointer_cast<const milvus::ArrayBoolFieldData>(
            CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(field_data)), 0, 5));
        EXPECT_THAT(field_data_ptr->Data(), ElementsAre(values.at(0), values.at(1)));

        field_data_ptr = std::dynamic_pointer_cast<const milvus::ArrayBoolFieldData>(
            CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(field_data)), 1, 2));
        EXPECT_THAT(field_data_ptr->Data(), ElementsAre(values.at(1)));
    }

    {
        auto values = std::vector<milvus::ArrayInt8FieldData::ElementT>{{2, 3}, {4}};
        milvus::ArrayInt8FieldData field_data{name, values};
        const auto field_data_ptr = std::dynamic_pointer_cast<const milvus::ArrayInt8FieldData>(
            CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(field_data)), 1, 2));
        EXPECT_THAT(field_data_ptr->Data(), ElementsAre(values.at(1)));
    }
    {
        auto values = std::vector<milvus::ArrayInt16FieldData::ElementT>{{2, 3}, {4}};
        milvus::ArrayInt16FieldData field_data{name, values};
        const auto field_data_ptr = std::dynamic_pointer_cast<const milvus::ArrayInt16FieldData>(
            CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(field_data)), 1, 2));
        EXPECT_THAT(field_data_ptr->Data(), ElementsAre(values.at(1)));
    }
    {
        auto values = std::vector<milvus::ArrayInt32FieldData::ElementT>{{2, 3}, {4}};
        milvus::ArrayInt32FieldData field_data{name, values};
        const auto field_data_ptr = std::dynamic_pointer_cast<const milvus::ArrayInt32FieldData>(
            CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(field_data)), 1, 2));
        EXPECT_THAT(field_data_ptr->Data(), ElementsAre(values.at(1)));
    }
    {
        auto values = std::vector<milvus::ArrayInt64FieldData::ElementT>{{2, 3}, {4}};
        milvus::ArrayInt64FieldData field_data{name, values};
        const auto field_data_ptr = std::dynamic_pointer_cast<const milvus::ArrayInt64FieldData>(
            CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(field_data)), 1, 2));
        EXPECT_THAT(field_data_ptr->Data(), ElementsAre(values.at(1)));
    }
    {
        auto values = std::vector<milvus::ArrayFloatFieldData::ElementT>{{0.2, 0.3}, {0.4}};
        milvus::ArrayFloatFieldData field_data{name, values};
        const auto field_data_ptr = std::dynamic_pointer_cast<const milvus::ArrayFloatFieldData>(
            CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(field_data)), 1, 2));
        EXPECT_THAT(field_data_ptr->Data(), ElementsAre(values.at(1)));
    }
    {
        auto values = std::vector<milvus::ArrayDoubleFieldData::ElementT>{{0.2, 0.3}, {0.4}};
        milvus::ArrayDoubleFieldData field_data{name, values};
        const auto field_data_ptr = std::dynamic_pointer_cast<const milvus::ArrayDoubleFieldData>(
            CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(field_data)), 1, 2));
        EXPECT_THAT(field_data_ptr->Data(), ElementsAre(values.at(1)));
    }
    {
        auto values = std::vector<milvus::ArrayVarCharFieldData::ElementT>{{"a", "bb"}, {"ccc"}};
        milvus::ArrayVarCharFieldData field_data{name, values};
        const auto field_data_ptr = std::dynamic_pointer_cast<const milvus::ArrayVarCharFieldData>(
            CreateMilvusFieldData(CreateProtoFieldData(static_cast<const milvus::Field&>(field_data)), 1, 2));
        EXPECT_THAT(field_data_ptr->Data(), ElementsAre(values.at(1)));
    }
}

TEST_F(DmlUtilsTest, IsInputFieldTest) {
    milvus::FieldSchema id_field{"foo", milvus::DataType::INT64, "foo", true, true};
    auto ret = milvus::IsInputField(id_field, true);
    EXPECT_TRUE(ret);
    ret = milvus::IsInputField(id_field, false);
    EXPECT_FALSE(ret);

    milvus::FieldSchema dummy_field{"foo", milvus::DataType::INT64, "foo", false, false};
    ret = milvus::IsInputField(dummy_field, true);
    EXPECT_TRUE(ret);
    ret = milvus::IsInputField(dummy_field, false);
    EXPECT_TRUE(ret);
}

TEST_F(DmlUtilsTest, CheckInsertInputTest) {
    auto createSchemaFunc = [](bool auto_id, bool dynamic_enabled) {
        milvus::CollectionSchema schema("my_coll");
        schema.SetEnableDynamicField(dynamic_enabled);
        schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, auto_id));
        schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(2));
        schema.AddField(milvus::FieldSchema("json", milvus::DataType::JSON));
        return std::move(schema);
    };

    milvus::CollectionDescPtr desc = std::make_shared<milvus::CollectionDesc>();
    desc->SetSchema(std::move(createSchemaFunc(true, false)));
    desc->SetID(1000);
    desc->SetDatabaseName("my_db");

    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::FloatVecFieldData>("vector", std::vector<std::vector<float>>{{1.0, 2.0}, {3.0, 4.0}}),
        std::make_shared<milvus::JSONFieldData>("json", std::vector<nlohmann::json>{{"age", 50}, {"age", 100}}),
    };

    {
        // auto-id is true, primary key field is not provided, insert is ok, upsert is wrong
        auto status = milvus::CheckInsertInput(desc, fields, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);

        status = milvus::CheckInsertInput(desc, fields, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

        // auto-id is true, primary key field is provided, insert is wrong, upsert is ok
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.emplace_back(std::move(std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{1, 2})));

        status = milvus::CheckInsertInput(desc, temp_fields, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

        status = milvus::CheckInsertInput(desc, temp_fields, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);
    }

    desc->SetSchema(std::move(createSchemaFunc(false, false)));
    {
        // auto-id is false, primary key field is not provided, insert is wrong, upsert is wrong
        auto status = milvus::CheckInsertInput(desc, fields, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

        status = milvus::CheckInsertInput(desc, fields, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

        // auto-id is false, primary key field is provided, insert is ok, upsert is ok
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.emplace_back(std::move(std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{1, 2})));

        status = milvus::CheckInsertInput(desc, temp_fields, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);

        status = milvus::CheckInsertInput(desc, temp_fields, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);
    }

    {
        // enable_dynamic_field is false, the dynamic field data is not json type, both insert and upsert are wrong
        auto dynamic_data = std::make_shared<milvus::Int64FieldData>(milvus::DYNAMIC_FIELD, std::vector<int64_t>{1, 2});
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.emplace_back(std::move(dynamic_data));

        auto status = milvus::CheckInsertInput(desc, temp_fields, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_AGUMENT);

        status = milvus::CheckInsertInput(desc, temp_fields, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_AGUMENT);
    }

    {
        // enable_dynamic_field is false, the dynamic field data is json type, both insert and upsert are wrong
        auto dynamic_data = std::make_shared<milvus::JSONFieldData>(
            milvus::DYNAMIC_FIELD, std::vector<nlohmann::json>{{"age", 50}, {"age", 100}});
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.emplace_back(std::move(dynamic_data));

        auto status = milvus::CheckInsertInput(desc, temp_fields, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

        status = milvus::CheckInsertInput(desc, temp_fields, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);
    }

    desc->SetSchema(std::move(createSchemaFunc(false, true)));
    {
        // enable_dynamic_field is true, the dynamic field data is not json type, both insert and upsert are wrong
        auto dummy_data = std::make_shared<milvus::Int64FieldData>(milvus::DYNAMIC_FIELD, std::vector<int64_t>{1, 2});
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.emplace_back(std::move(dummy_data));

        auto status = milvus::CheckInsertInput(desc, temp_fields, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_AGUMENT);

        status = milvus::CheckInsertInput(desc, temp_fields, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_AGUMENT);
    }

    {
        // enable_dynamic_field is true, the dynamic field data is json type, both insert and upsert are ok
        auto dummy_data = std::make_shared<milvus::JSONFieldData>(
            milvus::DYNAMIC_FIELD, std::vector<nlohmann::json>{{"age", 50}, {"age", 100}});
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.emplace_back(std::move(std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{1, 2})));
        temp_fields.emplace_back(std::move(dummy_data));

        auto status = milvus::CheckInsertInput(desc, temp_fields, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);

        status = milvus::CheckInsertInput(desc, temp_fields, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);
    }

    desc->SetSchema(std::move(createSchemaFunc(true, true)));
    {
        // enable_dynamic_field is true, no dynamic data provided
        // but field data missed
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.pop_back();

        auto status = milvus::CheckInsertInput(desc, temp_fields, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

        status = milvus::CheckInsertInput(desc, temp_fields, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);
    }
}
