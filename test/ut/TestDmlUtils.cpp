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
#include "milvus/utils/FP16.h"
#include "utils/Constants.h"
#include "utils/DmlUtils.h"
#include "utils/DqlUtils.h"
#include "utils/FieldDataSchema.h"

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class DmlUtilsTest : public ::testing::Test {};

namespace {
template <typename T, typename V>
void
VerifyCreateMilvusFieldData(std::vector<V> original_data) {
    milvus::FieldDataPtr field_ptr;
    std::shared_ptr<T> field_data = std::make_shared<T>("foo", original_data);
    milvus::FieldDataSchema bridge(field_data, nullptr);
    milvus::proto::schema::FieldData proto_data;
    auto status = milvus::CreateProtoFieldData(bridge, proto_data);
    EXPECT_TRUE(status.IsOk());
    status = milvus::CreateMilvusFieldData(proto_data, 1, 2, field_ptr);
    EXPECT_TRUE(status.IsOk());
    auto field_data_ptr = std::dynamic_pointer_cast<T>(field_ptr);
    EXPECT_THAT(field_data_ptr->Data(), ElementsAre(original_data.at(1), original_data.at(2)));

    status = milvus::CreateMilvusFieldData(proto_data, 10, 20, field_ptr);
    EXPECT_TRUE(status.IsOk());
    field_data_ptr = std::dynamic_pointer_cast<T>(field_ptr);
    EXPECT_TRUE(field_data_ptr->Data().empty());

    status = milvus::CreateMilvusFieldData(proto_data, 10, 10, field_ptr);
    EXPECT_TRUE(status.IsOk());
    field_data_ptr = std::dynamic_pointer_cast<T>(field_ptr);
    EXPECT_TRUE(field_data_ptr->Data().empty());

    status = milvus::CreateMilvusFieldData(proto_data, 0, 1, field_ptr);
    EXPECT_TRUE(status.IsOk());
    field_data_ptr = std::dynamic_pointer_cast<T>(field_ptr);
    EXPECT_THAT(field_data_ptr->Data(), ElementsAre(original_data.at(0)));

    status = milvus::CreateMilvusFieldData(proto_data, 0, original_data.size() + 1, field_ptr);
    EXPECT_TRUE(status.IsOk());
    field_data_ptr = std::dynamic_pointer_cast<T>(field_ptr);
    EXPECT_THAT(field_data_ptr->Data(), ElementsAreArray(original_data));
}
}  // namespace

TEST_F(DmlUtilsTest, IDArray) {
    milvus::proto::schema::IDs ids;
    ids.mutable_int_id()->add_data(10000);
    ids.mutable_int_id()->add_data(10001);
    auto id_array = milvus::CreateIDArray(ids);

    EXPECT_TRUE(id_array.IsIntegerID());
    EXPECT_THAT(id_array.IntIDArray(), ElementsAre(10000, 10001));

    ids.mutable_str_id()->add_data("10000");
    ids.mutable_str_id()->add_data("10001");
    id_array = milvus::CreateIDArray(ids);

    EXPECT_FALSE(id_array.IsIntegerID());
    EXPECT_THAT(id_array.StrIDArray(), ElementsAre("10000", "10001"));
}

TEST_F(DmlUtilsTest, CreateMilvusFieldDataWithRange_Scalar) {
    VerifyCreateMilvusFieldData<milvus::BoolFieldData, bool>(std::vector<bool>{false, true, false});
    VerifyCreateMilvusFieldData<milvus::Int8FieldData, int8_t>(std::vector<int8_t>{1, 2, 1});
    VerifyCreateMilvusFieldData<milvus::Int16FieldData, int16_t>(std::vector<int16_t>{6, 5, 2});
    VerifyCreateMilvusFieldData<milvus::Int32FieldData, int32_t>(std::vector<int32_t>{2, 3, 6});
    VerifyCreateMilvusFieldData<milvus::Int64FieldData, int64_t>(std::vector<int64_t>{9, 5, 7});
    VerifyCreateMilvusFieldData<milvus::FloatFieldData, float>(std::vector<float>{0.1, 0.2, 0.3});
    VerifyCreateMilvusFieldData<milvus::DoubleFieldData, double>(std::vector<double>{2.4, 3.4, 1.2});
    VerifyCreateMilvusFieldData<milvus::VarCharFieldData, std::string>(std::vector<std::string>{"a", "b", "c"});

    auto values =
        std::vector<nlohmann::json>{R"({"name":"aaa","age":18,"score":88})", R"({"name":"bbb","age":19,"score":99})",
                                    R"({"name":"ccc","age":15,"score":100})"};
    VerifyCreateMilvusFieldData<milvus::JSONFieldData, nlohmann::json>(values);
}

TEST_F(DmlUtilsTest, CreateMilvusFieldDataWithRange_Vector) {
    {
        auto values = std::vector<std::vector<uint8_t>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        VerifyCreateMilvusFieldData<milvus::BinaryVecFieldData, std::vector<uint8_t>>(values);
    }
    {
        auto values = std::vector<std::vector<float>>{{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}, {0.7f, 0.8f, 0.9f}};
        VerifyCreateMilvusFieldData<milvus::FloatVecFieldData, std::vector<float>>(values);

        std::vector<std::vector<uint16_t>> f16_values;
        for (auto vec : values) {
            f16_values.emplace_back(milvus::ArrayF32toF16(vec));
        }
        VerifyCreateMilvusFieldData<milvus::Float16VecFieldData, std::vector<uint16_t>>(f16_values);

        f16_values.clear();
        for (auto vec : values) {
            f16_values.emplace_back(milvus::ArrayF32toBF16(vec));
        }
        VerifyCreateMilvusFieldData<milvus::BFloat16VecFieldData, std::vector<uint16_t>>(f16_values);
    }
    {
        std::map<uint32_t, float> sparse1 = {{1, 0.1}, {2, 0.2}};
        std::map<uint32_t, float> sparse2 = {{66, 6.6}};
        std::map<uint32_t, float> sparse3 = {{99, 99.0}};
        auto values = std::vector<std::map<uint32_t, float>>{sparse1, sparse2, sparse3};
        VerifyCreateMilvusFieldData<milvus::SparseFloatVecFieldData, std::map<uint32_t, float>>(values);
    }
}

TEST_F(DmlUtilsTest, CreateMilvusFieldDataWithRange_Array) {
    {
        auto values = std::vector<milvus::ArrayBoolFieldData::ElementT>{{true, false}, {false}, {true, true}};
        VerifyCreateMilvusFieldData<milvus::ArrayBoolFieldData, milvus::ArrayBoolFieldData::ElementT>(values);
    }
    {
        auto values = std::vector<milvus::ArrayInt8FieldData::ElementT>{{2, 3}, {4}, {1, 0}};
        VerifyCreateMilvusFieldData<milvus::ArrayInt8FieldData, milvus::ArrayInt8FieldData::ElementT>(values);
    }
    {
        auto values = std::vector<milvus::ArrayInt16FieldData::ElementT>{{2, 3}, {4}, {}};
        VerifyCreateMilvusFieldData<milvus::ArrayInt16FieldData, milvus::ArrayInt16FieldData::ElementT>(values);
    }
    {
        auto values = std::vector<milvus::ArrayInt32FieldData::ElementT>{{2, 3}, {4}, {6}};
        VerifyCreateMilvusFieldData<milvus::ArrayInt32FieldData, milvus::ArrayInt32FieldData::ElementT>(values);
    }
    {
        auto values = std::vector<milvus::ArrayInt64FieldData::ElementT>{{2, 3}, {4}, {5, 6}};
        VerifyCreateMilvusFieldData<milvus::ArrayInt64FieldData, milvus::ArrayInt64FieldData::ElementT>(values);
    }
    {
        auto values = std::vector<milvus::ArrayFloatFieldData::ElementT>{{0.2, 0.3}, {0.4}, {5.5}};
        VerifyCreateMilvusFieldData<milvus::ArrayFloatFieldData, milvus::ArrayFloatFieldData::ElementT>(values);
    }
    {
        auto values = std::vector<milvus::ArrayDoubleFieldData::ElementT>{{0.2, 0.3}, {0.4}, {}};
        VerifyCreateMilvusFieldData<milvus::ArrayDoubleFieldData, milvus::ArrayDoubleFieldData::ElementT>(values);
    }
    {
        auto values = std::vector<milvus::ArrayVarCharFieldData::ElementT>{{"a", "bb"}, {"ccc"}, {}};
        VerifyCreateMilvusFieldData<milvus::ArrayVarCharFieldData, milvus::ArrayVarCharFieldData::ElementT>(values);
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
