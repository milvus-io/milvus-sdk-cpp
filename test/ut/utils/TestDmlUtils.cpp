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

#include <algorithm>
#include <cstring>

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

TEST_F(DmlUtilsTest, CreateMilvusFieldDataWithRangeScalar) {
    VerifyCreateMilvusFieldData<milvus::BoolFieldData, bool>(std::vector<bool>{false, true, false});
    VerifyCreateMilvusFieldData<milvus::Int8FieldData, int8_t>(std::vector<int8_t>{1, 2, 1});
    VerifyCreateMilvusFieldData<milvus::Int16FieldData, int16_t>(std::vector<int16_t>{6, 5, 2});
    VerifyCreateMilvusFieldData<milvus::Int32FieldData, int32_t>(std::vector<int32_t>{2, 3, 6});
    VerifyCreateMilvusFieldData<milvus::Int64FieldData, int64_t>(std::vector<int64_t>{9, 5, 7});
    VerifyCreateMilvusFieldData<milvus::FloatFieldData, float>(std::vector<float>{0.1, 0.2, 0.3});
    VerifyCreateMilvusFieldData<milvus::DoubleFieldData, double>(std::vector<double>{2.4, 3.4, 1.2});
    VerifyCreateMilvusFieldData<milvus::VarCharFieldData, std::string>(std::vector<std::string>{"a", "b", "c"});

    auto values = std::vector<nlohmann::json>{nlohmann::json::parse(R"({"name":"aaa","age":18,"score":88})"),
                                              nlohmann::json::parse(R"({"name":"bbb","age":19,"score":99})"),
                                              nlohmann::json::parse(R"({"name":"ccc","age":15,"score":100})")};
    VerifyCreateMilvusFieldData<milvus::JSONFieldData, nlohmann::json>(values);
}

TEST_F(DmlUtilsTest, CreateMilvusFieldDataWithRangeVector) {
    {
        auto values = std::vector<std::vector<uint8_t>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        VerifyCreateMilvusFieldData<milvus::BinaryVecFieldData, std::vector<uint8_t>>(values);
    }
    {
        auto values = std::vector<std::vector<float>>{{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}, {0.7f, 0.8f, 0.9f}};
        VerifyCreateMilvusFieldData<milvus::FloatVecFieldData, std::vector<float>>(values);

        std::vector<std::vector<uint16_t>> f16_values;
        f16_values.reserve(values.size());
        for (const auto& vec : values) {
            f16_values.push_back(milvus::ArrayF32toF16(vec));
        }
        VerifyCreateMilvusFieldData<milvus::Float16VecFieldData, std::vector<uint16_t>>(f16_values);

        f16_values.clear();
        for (const auto& vec : values) {
            f16_values.push_back(milvus::ArrayF32toBF16(vec));
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

TEST_F(DmlUtilsTest, CreateMilvusFieldDataWithRangeArray) {
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
        auto status = milvus::CheckInsertInput(desc, fields, false, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);

        status = milvus::CheckInsertInput(desc, fields, true, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

        status = milvus::CheckInsertInput(desc, fields, true, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);
        EXPECT_EQ(status.Message(), "Data is missed for field: pk");

        // explicit auto-id input is accepted for both insert and upsert
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.emplace_back(std::move(std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{1, 2})));

        status = milvus::CheckInsertInput(desc, temp_fields, false, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);

        status = milvus::CheckInsertInput(desc, temp_fields, true, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);

        temp_fields.back() = std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{1});
        status = milvus::CheckInsertInput(desc, temp_fields, true, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    }

    desc->SetSchema(std::move(createSchemaFunc(false, false)));
    {
        // auto-id is false, primary key field is not provided, insert is wrong, upsert is wrong
        auto status = milvus::CheckInsertInput(desc, fields, false, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

        status = milvus::CheckInsertInput(desc, fields, true, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

        // partial upsert still requires the primary key
        status = milvus::CheckInsertInput(desc, fields, true, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);
        EXPECT_EQ(status.Message(), "Data is missed for field: pk");

        // partial upsert allows non-primary fields to be omitted
        std::vector<milvus::FieldDataPtr> partial_fields{
            std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{1, 2}),
        };
        status = milvus::CheckInsertInput(desc, partial_fields, true, true);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);

        // auto-id is false, primary key field is provided, insert is ok, upsert is ok
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.emplace_back(std::move(std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{1, 2})));

        status = milvus::CheckInsertInput(desc, temp_fields, false, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);

        status = milvus::CheckInsertInput(desc, temp_fields, true, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);
    }

    {
        // enable_dynamic_field is false, the dynamic field data is not json type, both insert and upsert are wrong
        auto dynamic_data = std::make_shared<milvus::Int64FieldData>(milvus::DYNAMIC_FIELD, std::vector<int64_t>{1, 2});
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.emplace_back(std::move(dynamic_data));

        auto status = milvus::CheckInsertInput(desc, temp_fields, false, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);

        status = milvus::CheckInsertInput(desc, temp_fields, true, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    }

    {
        // enable_dynamic_field is false, the dynamic field data is json type, both insert and upsert are wrong
        auto dynamic_data = std::make_shared<milvus::JSONFieldData>(
            milvus::DYNAMIC_FIELD, std::vector<nlohmann::json>{{"age", 50}, {"age", 100}});
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.emplace_back(std::move(dynamic_data));

        auto status = milvus::CheckInsertInput(desc, temp_fields, false, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

        status = milvus::CheckInsertInput(desc, temp_fields, true, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);
    }

    desc->SetSchema(std::move(createSchemaFunc(false, true)));
    {
        // enable_dynamic_field is true, the dynamic field data is not json type, both insert and upsert are wrong
        auto dummy_data = std::make_shared<milvus::Int64FieldData>(milvus::DYNAMIC_FIELD, std::vector<int64_t>{1, 2});
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.emplace_back(std::move(dummy_data));

        auto status = milvus::CheckInsertInput(desc, temp_fields, false, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);

        status = milvus::CheckInsertInput(desc, temp_fields, true, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    }

    {
        // enable_dynamic_field is true, the dynamic field data is json type, both insert and upsert are ok
        auto dummy_data = std::make_shared<milvus::JSONFieldData>(
            milvus::DYNAMIC_FIELD, std::vector<nlohmann::json>{{"age", 50}, {"age", 100}});
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.emplace_back(std::move(std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{1, 2})));
        temp_fields.emplace_back(std::move(dummy_data));

        auto status = milvus::CheckInsertInput(desc, temp_fields, false, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);

        status = milvus::CheckInsertInput(desc, temp_fields, true, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::OK);
    }

    desc->SetSchema(std::move(createSchemaFunc(true, true)));
    {
        // enable_dynamic_field is true, no dynamic data provided
        // but field data missed
        std::vector<milvus::FieldDataPtr> temp_fields = fields;
        temp_fields.pop_back();

        auto status = milvus::CheckInsertInput(desc, temp_fields, false, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

        status = milvus::CheckInsertInput(desc, temp_fields, true, false);
        EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);
    }
}

TEST_F(DmlUtilsTest, CheckInsertInputFunctionOutputAndUnknownColumns) {
    milvus::CollectionSchema schema("test_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, false));
    schema.AddField(milvus::FieldSchema("text", milvus::DataType::VARCHAR).WithMaxLength(64));
    schema.AddField(milvus::FieldSchema("embedding", milvus::DataType::FLOAT_VECTOR).WithDimension(2));
    auto function = std::make_shared<milvus::Function>("embedding_func", milvus::FunctionType::TEXTEMBEDDING);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("embedding");
    schema.AddFunction(std::move(function));

    auto desc = std::make_shared<milvus::CollectionDesc>();
    desc->SetSchema(std::move(schema));

    std::vector<milvus::FieldDataPtr> columns{
        std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{1}),
        std::make_shared<milvus::VarCharFieldData>("text", std::vector<std::string>{"hello"}),
    };
    auto status = milvus::CheckInsertInput(desc, columns, false, false);
    EXPECT_TRUE(status.IsOk());
    status = milvus::CheckInsertInput(desc, columns, true, false);
    EXPECT_TRUE(status.IsOk());

    auto with_output = columns;
    with_output.emplace_back(
        std::make_shared<milvus::FloatVecFieldData>("embedding", std::vector<std::vector<float>>{{0.1f, 0.2f}}));
    status = milvus::CheckInsertInput(desc, with_output, false, false);
    EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);
    status = milvus::CheckInsertInput(desc, with_output, true, false);
    EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

    auto with_unknown = columns;
    with_unknown.emplace_back(std::make_shared<milvus::Int64FieldData>("unknown", std::vector<int64_t>{1}));
    status = milvus::CheckInsertInput(desc, with_unknown, false, false);
    EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);
    status = milvus::CheckInsertInput(desc, with_unknown, true, false);
    EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);
}

TEST_F(DmlUtilsTest, CheckInsertInputAllowsOmittedNullableAndDefaultColumns) {
    milvus::CollectionSchema schema("test_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, false));
    schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(2));
    schema.AddField(milvus::FieldSchema("name", milvus::DataType::VARCHAR).WithMaxLength(64).WithNullable(true));
    schema.AddField(milvus::FieldSchema("score", milvus::DataType::FLOAT).WithDefaultValue(1.0f));

    auto desc = std::make_shared<milvus::CollectionDesc>();
    desc->SetSchema(std::move(schema));
    std::vector<milvus::FieldDataPtr> columns{
        std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{1}),
        std::make_shared<milvus::FloatVecFieldData>("vector", std::vector<std::vector<float>>{{0.1f, 0.2f}}),
    };

    EXPECT_TRUE(milvus::CheckInsertInput(desc, columns, false, false).IsOk());
    EXPECT_TRUE(milvus::CheckInsertInput(desc, columns, true, false).IsOk());
}

TEST_F(DmlUtilsTest, EncodeSparseFloatVectorTest) {
    // empty sparse vector
    {
        std::map<uint32_t, float> sparse;
        auto encoded = milvus::EncodeSparseFloatVector(sparse);
        EXPECT_EQ(encoded.size(), 0u);
    }
    // single element
    {
        std::map<uint32_t, float> sparse = {{5, 1.5f}};
        auto encoded = milvus::EncodeSparseFloatVector(sparse);
        EXPECT_EQ(encoded.size(), 8u);

        // verify index bytes (little-endian uint32: 5)
        uint32_t index = 0;
        std::memcpy(&index, encoded.data(), sizeof(uint32_t));
        EXPECT_EQ(index, 5u);

        // verify value bytes (float: 1.5)
        float value = 0;
        std::memcpy(&value, encoded.data() + 4, sizeof(float));
        EXPECT_FLOAT_EQ(value, 1.5f);
    }
    // multiple elements: map is sorted by key
    {
        std::map<uint32_t, float> sparse = {{1, 0.1f}, {100, 0.2f}};
        auto encoded = milvus::EncodeSparseFloatVector(sparse);
        EXPECT_EQ(encoded.size(), 16u);

        uint32_t idx0 = 0, idx1 = 0;
        float val0 = 0, val1 = 0;
        std::memcpy(&idx0, encoded.data(), sizeof(uint32_t));
        std::memcpy(&val0, encoded.data() + 4, sizeof(float));
        std::memcpy(&idx1, encoded.data() + 8, sizeof(uint32_t));
        std::memcpy(&val1, encoded.data() + 12, sizeof(float));
        EXPECT_EQ(idx0, 1u);
        EXPECT_FLOAT_EQ(val0, 0.1f);
        EXPECT_EQ(idx1, 100u);
        EXPECT_FLOAT_EQ(val1, 0.2f);
    }
}

TEST_F(DmlUtilsTest, ParseSparseFloatVectorDictFormat) {
    // dict format: {"1": 0.1, "5": 0.2}
    {
        auto obj = nlohmann::json::parse(R"({"1": 0.1, "5": 0.2})");
        std::map<uint32_t, float> pairs;
        auto status = milvus::ParseSparseFloatVector(obj, "sparse_field", pairs);
        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(pairs.size(), 2u);
        EXPECT_FLOAT_EQ(pairs[1], 0.1f);
        EXPECT_FLOAT_EQ(pairs[5], 0.2f);
    }
}

TEST_F(DmlUtilsTest, ParseSparseFloatVectorIndicesValuesFormat) {
    // indices/values format
    {
        auto obj = nlohmann::json::parse(R"({"indices": [1, 5, 8], "values": [0.1, 0.2, 0.15]})");
        std::map<uint32_t, float> pairs;
        auto status = milvus::ParseSparseFloatVector(obj, "sparse_field", pairs);
        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(pairs.size(), 3u);
        EXPECT_FLOAT_EQ(pairs[1], 0.1f);
        EXPECT_FLOAT_EQ(pairs[5], 0.2f);
        EXPECT_FLOAT_EQ(pairs[8], 0.15f);
    }
}

TEST_F(DmlUtilsTest, ParseSparseFloatVectorInvalidInput) {
    // not an object
    {
        auto obj = nlohmann::json::parse(R"([1, 2, 3])");
        std::map<uint32_t, float> pairs;
        auto status = milvus::ParseSparseFloatVector(obj, "sparse_field", pairs);
        EXPECT_FALSE(status.IsOk());
    }
    // non-numeric values in indices/values format
    {
        auto obj = nlohmann::json::parse(R"({"indices": ["a"], "values": [0.1]})");
        std::map<uint32_t, float> pairs;
        auto status = milvus::ParseSparseFloatVector(obj, "sparse_field", pairs);
        EXPECT_FALSE(status.IsOk());
    }
    // non-array indices
    {
        auto obj = nlohmann::json::parse(R"({"indices": "bad", "values": "bad"})");
        std::map<uint32_t, float> pairs;
        auto status = milvus::ParseSparseFloatVector(obj, "sparse_field", pairs);
        EXPECT_FALSE(status.IsOk());
    }
}

TEST_F(DmlUtilsTest, IsRealFailureTest) {
    // success status is not a real failure
    {
        milvus::proto::common::Status status;
        status.set_code(0);
        EXPECT_FALSE(milvus::IsRealFailure(status));
    }
    // rate limit (code=8) is not a real failure
    {
        milvus::proto::common::Status status;
        status.set_code(8);
        EXPECT_FALSE(milvus::IsRealFailure(status));
    }
    // a real error code
    {
        milvus::proto::common::Status status;
        status.set_code(100);
        EXPECT_TRUE(milvus::IsRealFailure(status));
    }
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataSimple) {
    // Build a simple schema with pk (auto_id) + float vector
    milvus::CollectionSchema schema("test_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
    schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(2));
    schema.AddField(milvus::FieldSchema("name", milvus::DataType::VARCHAR));

    // valid row data
    milvus::EntityRows rows;
    rows.push_back(nlohmann::json::parse(R"({"vector": [1.0, 2.0], "name": "alice"})"));
    rows.push_back(nlohmann::json::parse(R"({"vector": [3.0, 4.0], "name": "bob"})"));

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());
    // Should have at least the vector and name fields
    EXPECT_GE(rpc_fields.size(), 2u);
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataDistinguishesInsertAndUpsert) {
    milvus::CollectionSchema schema("test_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
    schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(2));

    milvus::EntityRows rows{nlohmann::json{{"vector", {1.0, 2.0}}}};

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());

    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(rows, schema, true, false, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);

    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(rows, schema, true, true, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "The field: pk is not provided.");
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataExplicitAutoIDInsert) {
    milvus::CollectionSchema schema("test_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
    schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(2));

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    milvus::EntityRows rows{
        nlohmann::json{{"pk", 1}, {"vector", {0.1f, 0.2f}}},
        nlohmann::json{{"pk", 2}, {"vector", {0.3f, 0.4f}}},
    };
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());

    rpc_fields.clear();
    rows = {
        nlohmann::json{{"pk", 1}, {"vector", {0.1f, 0.2f}}},
        nlohmann::json{{"vector", {0.3f, 0.4f}}},
    };
    status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "The row count of input fields is inconsistent");

    rpc_fields.clear();
    rows = {
        nlohmann::json{{"vector", {0.1f, 0.2f}}},
        nlohmann::json{{"pk", 2}, {"vector", {0.3f, 0.4f}}},
    };
    status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "The row count of input fields is inconsistent");
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataRejectsUnexpectedFields) {
    milvus::CollectionSchema schema("test_coll");
    schema.SetEnableDynamicField(false);
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
    schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(2));
    schema.AddField(milvus::FieldSchema("embedding", milvus::DataType::FLOAT_VECTOR).WithDimension(2));
    auto function = std::make_shared<milvus::Function>("embedding_func", milvus::FunctionType::TEXTEMBEDDING);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("embedding");
    schema.AddFunction(function);

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    milvus::EntityRows rows{nlohmann::json{{"vector", {1.0, 2.0}}, {"embedding", {3.0, 4.0}}}};
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(rows, schema, true, false, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

    rpc_fields.clear();
    rows = {nlohmann::json{{"vector", {1.0, 2.0}}, {"unknown", 10}}};
    status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(rows, schema, true, false, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataSignalsStaleFunctionOutputSchema) {
    milvus::CollectionSchema stale_schema("test_coll");
    stale_schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, false));
    stale_schema.AddField(milvus::FieldSchema("text", milvus::DataType::VARCHAR).WithMaxLength(64));
    stale_schema.AddField(milvus::FieldSchema("embedding", milvus::DataType::FLOAT_VECTOR).WithDimension(2));
    auto function = std::make_shared<milvus::Function>("embedding_func", milvus::FunctionType::TEXTEMBEDDING);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("embedding");
    stale_schema.AddFunction(function);

    milvus::EntityRows rows{
        nlohmann::json{{"pk", 1}, {"text", "hello"}, {"embedding", {0.1f, 0.2f}}},
    };
    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    auto status = milvus::CheckAndSetRowData(rows, stale_schema, false, false, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);
    status = milvus::CheckAndSetRowData(rows, stale_schema, true, false, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::DATA_UNMATCH_SCHEMA);

    milvus::CollectionSchema refreshed_schema("test_coll");
    refreshed_schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, false));
    refreshed_schema.AddField(milvus::FieldSchema("text", milvus::DataType::VARCHAR).WithMaxLength(64));
    refreshed_schema.AddField(milvus::FieldSchema("embedding", milvus::DataType::FLOAT_VECTOR).WithDimension(2));

    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(rows, refreshed_schema, false, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());
    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(rows, refreshed_schema, true, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataDynamicFields) {
    milvus::CollectionSchema schema("test_coll");
    schema.SetEnableDynamicField(true);
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, false));
    schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(2));

    milvus::EntityRows rows{
        nlohmann::json{{"pk", 1}, {"vector", {0.1f, 0.2f}}, {"color", "red"}},
    };
    auto expect_dynamic = [&schema, &rows](bool is_upsert) {
        std::vector<milvus::proto::schema::FieldData> rpc_fields;
        auto status = milvus::CheckAndSetRowData(rows, schema, is_upsert, false, rpc_fields);
        EXPECT_TRUE(status.IsOk());
        auto it = std::find_if(rpc_fields.begin(), rpc_fields.end(), [](const auto& field) {
            return field.is_dynamic() && field.field_name() == milvus::DYNAMIC_FIELD;
        });
        ASSERT_NE(it, rpc_fields.end());
        ASSERT_EQ(it->scalars().json_data().data_size(), 1);
        const nlohmann::json expected{{"color", "red"}};
        EXPECT_EQ(nlohmann::json::parse(it->scalars().json_data().data(0)), expected);
    };

    expect_dynamic(false);
    expect_dynamic(true);
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataStructValidation) {
    milvus::CollectionSchema schema("test_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, false));
    schema.AddField(milvus::FieldSchema("tags", milvus::DataType::ARRAY)
                        .WithElementType(milvus::DataType::INT32)
                        .WithMaxCapacity(10));
    milvus::StructFieldSchema struct_field("structs");
    struct_field.SetMaxCapacity(10);
    struct_field.AddField(milvus::FieldSchema("values", milvus::DataType::ARRAY)
                              .WithElementType(milvus::DataType::INT32)
                              .WithMaxCapacity(10));
    schema.AddStructField(std::move(struct_field));

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    milvus::EntityRows rows{nlohmann::json{{"pk", 1}, {"tags", {1, 2}}}};
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "The struct field: structs is not provided.");

    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(rows, schema, true, false, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "The struct field: structs is not provided.");

    rows = {nlohmann::json{{"pk", 1}, {"tags", {1, 2}}}};
    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(rows, schema, true, true, rpc_fields);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(std::count_if(rpc_fields.begin(), rpc_fields.end(),
                            [](const auto& field) { return field.field_name() == "structs"; }),
              0);

    rows = {nlohmann::json{{"pk", 1}, {"tags", {1, 2}}, {"structs", {{{"values", {3, 4}}}}}}};
    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(std::count_if(rpc_fields.begin(), rpc_fields.end(),
                            [](const auto& field) { return field.field_name() == "structs"; }),
              1);

    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(rows, schema, true, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());

    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(rows, schema, true, true, rpc_fields);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(std::count_if(rpc_fields.begin(), rpc_fields.end(),
                            [](const auto& field) { return field.field_name() == "structs"; }),
              1);

    rows = {nlohmann::json{{"pk", 1}, {"structs[values]", {3, 4}}}};
    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(rows, schema, true, true, rpc_fields);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "Partial struct update is not supported for struct sub-field: structs[values]");
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataRejectsInconsistentPartialUpsertRows) {
    milvus::CollectionSchema schema("test_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, false));
    schema.AddField(milvus::FieldSchema("tags", milvus::DataType::ARRAY)
                        .WithElementType(milvus::DataType::INT32)
                        .WithMaxCapacity(10));
    schema.AddField(milvus::FieldSchema("score", milvus::DataType::INT32));

    auto expect_inconsistent = [&schema](milvus::EntityRows rows) {
        std::vector<milvus::proto::schema::FieldData> rpc_fields;
        auto status = milvus::CheckAndSetRowData(rows, schema, true, true, rpc_fields);
        EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
        EXPECT_EQ(status.Message(), "The row count of partial update fields is inconsistent");
    };

    expect_inconsistent({
        nlohmann::json{{"pk", 1}, {"tags", {1, 2}}},
        nlohmann::json{{"pk", 2}},
    });
    expect_inconsistent({
        nlohmann::json{{"pk", 1}, {"tags", {1, 2}}},
        nlohmann::json{{"pk", 2}, {"score", 10}},
    });
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataInvalidRow) {
    milvus::CollectionSchema schema("test_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
    schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(2));

    // non-object row should fail
    milvus::EntityRows rows;
    rows.push_back(nlohmann::json::parse(R"([1, 2])"));

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_FALSE(status.IsOk());
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataAllScalarTypes) {
    // schema with all scalar types to exercise CheckAndSetScalar for each type
    milvus::CollectionSchema schema("test_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
    schema.AddField(milvus::FieldSchema("f_bool", milvus::DataType::BOOL));
    schema.AddField(milvus::FieldSchema("f_int8", milvus::DataType::INT8));
    schema.AddField(milvus::FieldSchema("f_int16", milvus::DataType::INT16));
    schema.AddField(milvus::FieldSchema("f_int32", milvus::DataType::INT32));
    schema.AddField(milvus::FieldSchema("f_int64", milvus::DataType::INT64));
    schema.AddField(milvus::FieldSchema("f_float", milvus::DataType::FLOAT));
    schema.AddField(milvus::FieldSchema("f_double", milvus::DataType::DOUBLE));
    schema.AddField(milvus::FieldSchema("f_varchar", milvus::DataType::VARCHAR).WithMaxLength(128));
    schema.AddField(milvus::FieldSchema("f_json", milvus::DataType::JSON));
    schema.AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR).WithDimension(2));

    milvus::EntityRows rows;
    rows.push_back(nlohmann::json{
        {"f_bool", true},
        {"f_int8", 1},
        {"f_int16", 100},
        {"f_int32", 10000},
        {"f_int64", 100000},
        {"f_float", 1.5},
        {"f_double", 2.5},
        {"f_varchar", "hello"},
        {"f_json", nlohmann::json{{"key", "val"}}},
        {"vec", {0.1, 0.2}},
    });

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());
    EXPECT_GE(rpc_fields.size(), 10u);
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataVectorTypes) {
    // test binary vector, sparse vector, float16 vector via row-based insert
    {
        milvus::CollectionSchema schema("bin_coll");
        schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
        schema.AddField(milvus::FieldSchema("bin_vec", milvus::DataType::BINARY_VECTOR).WithDimension(32));

        milvus::EntityRows rows;
        // 32-dim binary = 4 bytes as uint8 array
        rows.push_back(nlohmann::json{{"bin_vec", {255, 0, 171, 205}}});

        std::vector<milvus::proto::schema::FieldData> rpc_fields;
        auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
        EXPECT_TRUE(status.IsOk());
    }

    {
        milvus::CollectionSchema schema("sparse_coll");
        schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
        schema.AddField(milvus::FieldSchema("sp_vec", milvus::DataType::SPARSE_FLOAT_VECTOR));

        milvus::EntityRows rows;
        rows.push_back(nlohmann::json{{"sp_vec", nlohmann::json{{"1", 0.5}, {"3", 0.8}}}});

        std::vector<milvus::proto::schema::FieldData> rpc_fields;
        auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
        EXPECT_TRUE(status.IsOk());
    }

    {
        milvus::CollectionSchema schema("fp16_coll");
        schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
        schema.AddField(milvus::FieldSchema("fp16_vec", milvus::DataType::FLOAT16_VECTOR).WithDimension(2));

        milvus::EntityRows rows;
        rows.push_back(nlohmann::json{{"fp16_vec", {1.0, 2.0}}});

        std::vector<milvus::proto::schema::FieldData> rpc_fields;
        auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
        EXPECT_TRUE(status.IsOk());
    }

    {
        milvus::CollectionSchema schema("int8_coll");
        schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
        schema.AddField(milvus::FieldSchema("i8_vec", milvus::DataType::INT8_VECTOR).WithDimension(3));

        milvus::EntityRows rows;
        rows.push_back(nlohmann::json{{"i8_vec", {1, -2, 3}}});

        std::vector<milvus::proto::schema::FieldData> rpc_fields;
        auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
        EXPECT_TRUE(status.IsOk());
    }
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataArrayField) {
    milvus::CollectionSchema schema("arr_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
    schema.AddField(milvus::FieldSchema("arr", milvus::DataType::ARRAY)
                        .WithElementType(milvus::DataType::INT32)
                        .WithMaxCapacity(10));
    schema.AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR).WithDimension(2));

    milvus::EntityRows rows;
    rows.push_back(nlohmann::json{{"arr", {10, 20, 30}}, {"vec", {0.1, 0.2}}});
    rows.push_back(nlohmann::json{{"arr", {40}}, {"vec", {0.3, 0.4}}});

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataNullableField) {
    milvus::CollectionSchema schema("nullable_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
    schema.AddField(milvus::FieldSchema("name", milvus::DataType::VARCHAR).WithMaxLength(64).WithNullable(true));
    schema.AddField(milvus::FieldSchema("age", milvus::DataType::INT8).WithNullable(true));
    schema.AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR).WithDimension(2));

    milvus::EntityRows rows;
    // row with values
    rows.push_back(nlohmann::json{{"name", "Alice"}, {"age", 25}, {"vec", {0.1, 0.2}}});
    // row with null values
    rows.push_back(nlohmann::json{{"name", nullptr}, {"age", nullptr}, {"vec", {0.3, 0.4}}});
    // row with omitted nullable fields
    rows.push_back(nlohmann::json{{"vec", {0.5, 0.6}}});

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());

    milvus::EntityRows upsert_rows{
        nlohmann::json{{"pk", 1}, {"vec", {0.5, 0.6}}},
    };
    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(upsert_rows, schema, true, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataDefaultValue) {
    milvus::CollectionSchema schema("default_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
    schema.AddField(milvus::FieldSchema("score", milvus::DataType::FLOAT).WithDefaultValue(9.99));
    schema.AddField(milvus::FieldSchema("label", milvus::DataType::VARCHAR).WithMaxLength(64).WithDefaultValue("none"));
    schema.AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR).WithDimension(2));

    milvus::EntityRows rows;
    // row with explicit values
    rows.push_back(nlohmann::json{{"score", 5.0}, {"label", "good"}, {"vec", {0.1, 0.2}}});
    // row with omitted fields (should use defaults)
    rows.push_back(nlohmann::json{{"vec", {0.3, 0.4}}});

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());

    milvus::EntityRows upsert_rows{
        nlohmann::json{{"pk", 1}, {"vec", {0.3, 0.4}}},
    };
    rpc_fields.clear();
    status = milvus::CheckAndSetRowData(upsert_rows, schema, true, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataGeometryAndTimestamptz) {
    milvus::CollectionSchema schema("geo_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
    schema.AddField(milvus::FieldSchema("geo", milvus::DataType::GEOMETRY));
    schema.AddField(milvus::FieldSchema("tsz", milvus::DataType::TIMESTAMPTZ));
    schema.AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR).WithDimension(2));

    milvus::EntityRows rows;
    rows.push_back(nlohmann::json{{"geo", "POINT (1 1)"}, {"tsz", "2025-01-01T00:00:00+00:00"}, {"vec", {0.1, 0.2}}});

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(DmlUtilsTest, CreateProtoFieldDataWithNullable) {
    // test CreateProtoFieldData + CreateMilvusFieldData roundtrip with nullable fields
    // this exercises CopyValidData template instantiations
    auto field_data = std::make_shared<milvus::VarCharFieldData>("name");
    field_data->Add("Alice");
    field_data->AddNull();
    field_data->Add("Charlie");

    milvus::FieldDataSchema bridge(field_data, nullptr);
    milvus::proto::schema::FieldData proto_data;
    auto status = milvus::CreateProtoFieldData(bridge, proto_data);
    EXPECT_TRUE(status.IsOk());

    milvus::FieldDataPtr result;
    status = milvus::CreateMilvusFieldData(proto_data, result);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(result->Count(), 3);

    auto typed = std::dynamic_pointer_cast<milvus::VarCharFieldData>(result);
    EXPECT_NE(typed, nullptr);
    EXPECT_EQ(typed->Value(0), "Alice");
    EXPECT_EQ(typed->Value(2), "Charlie");
}

TEST_F(DmlUtilsTest, CreateProtoFieldDataWithNullableInt) {
    auto field_data = std::make_shared<milvus::Int16FieldData>("age");
    field_data->Add(10);
    field_data->AddNull();
    field_data->Add(30);

    milvus::FieldDataSchema bridge(field_data, nullptr);
    milvus::proto::schema::FieldData proto_data;
    auto status = milvus::CreateProtoFieldData(bridge, proto_data);
    EXPECT_TRUE(status.IsOk());

    milvus::FieldDataPtr result;
    status = milvus::CreateMilvusFieldData(proto_data, result);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataNullableVector) {
    milvus::CollectionSchema schema("nullable_vector_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
    schema.AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR).WithDimension(2).WithNullable(true));

    milvus::EntityRows rows;
    rows.push_back(nlohmann::json{{"vec", {0.1, 0.2}}});
    rows.push_back(nlohmann::json{{"vec", nullptr}});
    rows.push_back(nlohmann::json::object());
    rows.push_back(nlohmann::json{{"vec", {0.3, 0.4}}});

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    ASSERT_TRUE(status.IsOk()) << status.Message();

    auto it = std::find_if(rpc_fields.begin(), rpc_fields.end(),
                           [](const auto& field) { return field.field_name() == "vec"; });
    ASSERT_NE(it, rpc_fields.end());
    EXPECT_THAT(it->valid_data(), ElementsAre(true, false, false, true));
    EXPECT_EQ(it->vectors().dim(), 2);
    EXPECT_THAT(it->vectors().float_vector().data(), ElementsAre(0.1f, 0.2f, 0.3f, 0.4f));
}

TEST_F(DmlUtilsTest, CheckAndSetRowDataAllNullVectorPreservesDimension) {
    milvus::CollectionSchema schema("nullable_vector_coll");
    schema.AddField(milvus::FieldSchema("pk", milvus::DataType::INT64, "pk", true, true));
    schema.AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR).WithDimension(2).WithNullable(true));

    milvus::EntityRows rows;
    rows.push_back(nlohmann::json{{"vec", nullptr}});
    rows.push_back(nlohmann::json::object());

    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    auto status = milvus::CheckAndSetRowData(rows, schema, false, false, rpc_fields);
    ASSERT_TRUE(status.IsOk()) << status.Message();

    auto it = std::find_if(rpc_fields.begin(), rpc_fields.end(),
                           [](const auto& field) { return field.field_name() == "vec"; });
    ASSERT_NE(it, rpc_fields.end());
    EXPECT_THAT(it->valid_data(), ElementsAre(false, false));
    EXPECT_EQ(it->vectors().dim(), 2);
    EXPECT_TRUE(it->vectors().float_vector().data().empty());
}

TEST_F(DmlUtilsTest, NullableVectorColumnRoundTrip) {
    milvus::CollectionSchema schema("nullable_vector_coll");
    schema.AddField(milvus::FieldSchema("binary", milvus::DataType::BINARY_VECTOR).WithDimension(8).WithNullable(true));
    schema.AddField(milvus::FieldSchema("float", milvus::DataType::FLOAT_VECTOR).WithDimension(2).WithNullable(true));
    schema.AddField(milvus::FieldSchema("sparse", milvus::DataType::SPARSE_FLOAT_VECTOR).WithNullable(true));
    schema.AddField(
        milvus::FieldSchema("float16", milvus::DataType::FLOAT16_VECTOR).WithDimension(2).WithNullable(true));
    schema.AddField(
        milvus::FieldSchema("bfloat16", milvus::DataType::BFLOAT16_VECTOR).WithDimension(2).WithNullable(true));
    schema.AddField(milvus::FieldSchema("int8", milvus::DataType::INT8_VECTOR).WithDimension(2).WithNullable(true));

    auto binary = std::make_shared<milvus::BinaryVecFieldData>("binary");
    binary->AddNull();
    EXPECT_EQ(binary->Add({1}), milvus::StatusCode::OK);
    binary->AddNull();
    EXPECT_EQ(binary->Add({2}), milvus::StatusCode::OK);
    auto floats = std::make_shared<milvus::FloatVecFieldData>("float");
    floats->AddNull();
    EXPECT_EQ(floats->Add({0.1f, 0.2f}), milvus::StatusCode::OK);
    floats->AddNull();
    EXPECT_EQ(floats->Add({0.3f, 0.4f}), milvus::StatusCode::OK);
    auto sparse = std::make_shared<milvus::SparseFloatVecFieldData>("sparse");
    sparse->AddNull();
    EXPECT_EQ(sparse->Add({{1, 0.1f}}), milvus::StatusCode::OK);
    sparse->AddNull();
    EXPECT_EQ(sparse->Add({{2, 0.2f}}), milvus::StatusCode::OK);
    auto float16 = std::make_shared<milvus::Float16VecFieldData>("float16");
    float16->AddNull();
    EXPECT_EQ(float16->Add({1, 2}), milvus::StatusCode::OK);
    float16->AddNull();
    EXPECT_EQ(float16->Add({3, 4}), milvus::StatusCode::OK);
    auto bfloat16 = std::make_shared<milvus::BFloat16VecFieldData>("bfloat16");
    bfloat16->AddNull();
    EXPECT_EQ(bfloat16->Add({1, 2}), milvus::StatusCode::OK);
    bfloat16->AddNull();
    EXPECT_EQ(bfloat16->Add({3, 4}), milvus::StatusCode::OK);
    auto int8 = std::make_shared<milvus::Int8VecFieldData>("int8");
    int8->AddNull();
    EXPECT_EQ(int8->Add({1, 2}), milvus::StatusCode::OK);
    int8->AddNull();
    EXPECT_EQ(int8->Add({3, 4}), milvus::StatusCode::OK);

    std::vector<milvus::FieldDataPtr> columns{binary, floats, sparse, float16, bfloat16, int8};
    std::vector<milvus::proto::schema::FieldData> rpc_fields;
    auto status = milvus::CreateProtoFieldDatas(schema, columns, rpc_fields);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    ASSERT_EQ(rpc_fields.size(), columns.size());

    for (const auto& proto_field : rpc_fields) {
        EXPECT_THAT(proto_field.valid_data(), ElementsAre(false, true, false, true));
        milvus::FieldDataPtr result;
        status = milvus::CreateMilvusFieldData(proto_field, result);
        ASSERT_TRUE(status.IsOk()) << status.Message();
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->Count(), 4);
    }

    auto float_proto = std::find_if(rpc_fields.begin(), rpc_fields.end(),
                                    [](const auto& field) { return field.field_name() == "float"; });
    ASSERT_NE(float_proto, rpc_fields.end());
    EXPECT_EQ(float_proto->vectors().dim(), 2);
    EXPECT_THAT(float_proto->vectors().float_vector().data(), ElementsAre(0.1f, 0.2f, 0.3f, 0.4f));

    milvus::FieldDataPtr sliced;
    status = milvus::CreateMilvusFieldData(*float_proto, 1, 2, sliced);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    auto sliced_float = std::dynamic_pointer_cast<milvus::FloatVecFieldData>(sliced);
    ASSERT_NE(sliced_float, nullptr);
    ASSERT_EQ(sliced_float->Count(), 2);
    EXPECT_FALSE(sliced_float->IsNull(0));
    EXPECT_THAT(sliced_float->Value(0), ElementsAre(0.1f, 0.2f));
    EXPECT_TRUE(sliced_float->IsNull(1));
}

TEST_F(DmlUtilsTest, NullableVectorColumnAllNullPreservesDimension) {
    auto field = std::make_shared<milvus::FloatVecFieldData>("float");
    field->AddNull();
    field->AddNull();
    auto schema = std::make_shared<milvus::FieldSchema>(
        milvus::FieldSchema("float", milvus::DataType::FLOAT_VECTOR).WithDimension(4).WithNullable(true));

    milvus::FieldDataSchema bridge(field, schema);
    milvus::proto::schema::FieldData proto_data;
    auto status = milvus::CreateProtoFieldData(bridge, proto_data);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    EXPECT_THAT(proto_data.valid_data(), ElementsAre(false, false));
    EXPECT_EQ(proto_data.vectors().dim(), 4);
    EXPECT_TRUE(proto_data.vectors().float_vector().data().empty());
    EXPECT_EQ(proto_data.vectors().float_vector().data().Capacity(), 0);

    milvus::FieldDataPtr result;
    status = milvus::CreateMilvusFieldData(proto_data, result);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    auto result_float = std::dynamic_pointer_cast<milvus::FloatVecFieldData>(result);
    ASSERT_NE(result_float, nullptr);
    ASSERT_EQ(result_float->Count(), 2);
    EXPECT_TRUE(result_float->IsNull(0));
    EXPECT_TRUE(result_float->IsNull(1));
}

TEST_F(DmlUtilsTest, NullableVectorColumnRejectsMismatchedValidData) {
    auto field = std::make_shared<milvus::FloatVecFieldData>(
        "float", std::vector<std::vector<float>>{{0.1f, 0.2f}, {0.3f, 0.4f}}, std::vector<bool>{true});
    auto schema = std::make_shared<milvus::FieldSchema>(
        milvus::FieldSchema("float", milvus::DataType::FLOAT_VECTOR).WithDimension(2).WithNullable(true));

    milvus::FieldDataSchema bridge(field, schema);
    milvus::proto::schema::FieldData proto_data;
    auto status = milvus::CreateProtoFieldData(bridge, proto_data);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "The valid data count does not match the row count for field: float");
}

TEST_F(DmlUtilsTest, DenseVectorColumnsRejectRaggedRows) {
    auto verify = [](const milvus::FieldDataPtr& field, milvus::DataType type, int64_t dimension) {
        auto schema =
            std::make_shared<milvus::FieldSchema>(milvus::FieldSchema(field->Name(), type).WithDimension(dimension));
        milvus::FieldDataSchema bridge(field, schema);
        milvus::proto::schema::FieldData proto_data;
        auto status = milvus::CreateProtoFieldData(bridge, proto_data);
        EXPECT_FALSE(status.IsOk());
        EXPECT_EQ(status.Code(), milvus::StatusCode::DIMENSION_NOT_EQUAL);
    };

    verify(std::make_shared<milvus::BinaryVecFieldData>("binary", std::vector<std::vector<uint8_t>>{{1}, {2, 3, 4}}),
           milvus::DataType::BINARY_VECTOR, 16);
    verify(std::make_shared<milvus::FloatVecFieldData>("float",
                                                       std::vector<std::vector<float>>{{1.0f}, {2.0f, 3.0f, 4.0f}}),
           milvus::DataType::FLOAT_VECTOR, 2);
    verify(std::make_shared<milvus::Float16VecFieldData>("float16", std::vector<std::vector<uint16_t>>{{1}, {2, 3, 4}}),
           milvus::DataType::FLOAT16_VECTOR, 2);
    verify(
        std::make_shared<milvus::BFloat16VecFieldData>("bfloat16", std::vector<std::vector<uint16_t>>{{1}, {2, 3, 4}}),
        milvus::DataType::BFLOAT16_VECTOR, 2);
    verify(std::make_shared<milvus::Int8VecFieldData>("int8", std::vector<std::vector<int8_t>>{{1}, {2, 3, 4}}),
           milvus::DataType::INT8_VECTOR, 2);
}
