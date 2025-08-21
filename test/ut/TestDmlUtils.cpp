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
#include "utils/GtsDict.h"

class DmlUtilsTest : public ::testing::Test {};

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

TEST_F(DmlUtilsTest, DeduceGuaranteeTimestampTest) {
    auto ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::NONE, "db", "coll");
    EXPECT_EQ(ts, 1);

    ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::SESSION, "db", "coll");
    EXPECT_EQ(ts, 1);

    milvus::GtsDict::GetInstance().UpdateCollectionTs("db", "coll", 999);
    ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::NONE, "db", "coll");
    EXPECT_EQ(ts, 999);

    ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::SESSION, "db", "coll");
    EXPECT_EQ(ts, 999);

    ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::STRONG, "db", "coll");
    EXPECT_EQ(ts, milvus::GuaranteeStrongTs());

    ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::BOUNDED, "db", "coll");
    EXPECT_EQ(ts, 2);

    ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::EVENTUALLY, "db", "coll");
    EXPECT_EQ(ts, 1);
}
