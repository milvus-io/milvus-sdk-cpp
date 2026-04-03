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

#include "milvus/types/Constants.h"
#include "milvus/types/FieldSchema.h"

class FieldSchemaTest : public ::testing::Test {};

TEST_F(FieldSchemaTest, GeneralTesting) {
    std::string name = "f0";
    std::string desc = "desc";
    milvus::DataType dt = milvus::DataType::DOUBLE;
    bool is_primary_key = true;
    bool auto_id = false;
    std::map<std::string, std::string> params;
    params.insert(std::make_pair("dummy", "dummy"));

    milvus::FieldSchema schema;
    EXPECT_EQ(schema.Dimension(), 0);
    schema.SetName(name);
    schema.SetDescription(desc);
    schema.SetDataType(dt);
    schema.SetPrimaryKey(is_primary_key);
    schema.SetAutoID(auto_id);
    schema.SetTypeParams(params);
    schema.SetTypeParams(std::move(params));
    EXPECT_TRUE(schema.SetDimension(256));
    EXPECT_FALSE(schema.SetDimension(0));

    EXPECT_EQ(name, schema.Name());
    EXPECT_EQ(desc, schema.Description());
    EXPECT_EQ(dt, schema.FieldDataType());
    EXPECT_EQ(is_primary_key, schema.IsPrimaryKey());
    EXPECT_EQ(auto_id, schema.AutoID());

    auto& type_params = schema.TypeParams();
    EXPECT_TRUE(type_params.find(milvus::DIM) != type_params.end());
    EXPECT_EQ("256", type_params.at(milvus::DIM));
}

TEST_F(FieldSchemaTest, TestWithDimention) {
    EXPECT_EQ("1024", milvus::FieldSchema("vectors", milvus::DataType::FLOAT_VECTOR, "")
                          .WithDimension(1024)
                          .TypeParams()
                          .at(milvus::DIM));
}

TEST_F(FieldSchemaTest, TestForMaxLength) {
    auto schema = milvus::FieldSchema("name", milvus::DataType::VARCHAR, "");
    EXPECT_EQ(65535, schema.MaxLength());
    schema = milvus::FieldSchema("name", milvus::DataType::VARCHAR, "").WithMaxLength(200);
    EXPECT_EQ("200", schema.TypeParams().at(milvus::MAX_LENGTH));
    schema.SetMaxLength(300);
    EXPECT_EQ("300", schema.TypeParams().at(milvus::MAX_LENGTH));
    EXPECT_EQ(300, schema.MaxLength());
}

TEST_F(FieldSchemaTest, TestWithCapacity) {
    auto schema = milvus::FieldSchema("array", milvus::DataType::ARRAY, "").WithMaxCapacity(100);
    EXPECT_EQ(100, schema.MaxCapacity());
    EXPECT_EQ("100", schema.TypeParams().at(milvus::MAX_CAPACITY));
}

TEST_F(FieldSchemaTest, TestElementType) {
    auto schema = milvus::FieldSchema("array", milvus::DataType::ARRAY, "").WithElementType(milvus::DataType::INT16);
    EXPECT_EQ(milvus::DataType::INT16, schema.ElementType());
}

TEST_F(FieldSchemaTest, WithName) {
    auto schema = milvus::FieldSchema().WithName("myfield");
    EXPECT_EQ("myfield", schema.Name());
}

TEST_F(FieldSchemaTest, SetAndWithDescription) {
    milvus::FieldSchema schema;
    schema.SetDescription("desc1");
    EXPECT_EQ("desc1", schema.Description());

    auto& ref = schema.WithDescription("desc2");
    EXPECT_EQ("desc2", ref.Description());
}

TEST_F(FieldSchemaTest, WithDataType) {
    auto schema = milvus::FieldSchema().WithDataType(milvus::DataType::FLOAT);
    EXPECT_EQ(milvus::DataType::FLOAT, schema.FieldDataType());
}

TEST_F(FieldSchemaTest, SetElementType) {
    milvus::FieldSchema schema("arr", milvus::DataType::ARRAY, "");
    schema.SetElementType(milvus::DataType::INT32);
    EXPECT_EQ(milvus::DataType::INT32, schema.ElementType());
}

TEST_F(FieldSchemaTest, WithPrimaryKey) {
    auto schema = milvus::FieldSchema().WithPrimaryKey(true);
    EXPECT_TRUE(schema.IsPrimaryKey());

    auto schema2 = schema.WithPrimaryKey(false);
    EXPECT_FALSE(schema2.IsPrimaryKey());
}

TEST_F(FieldSchemaTest, WithAutoID) {
    auto schema = milvus::FieldSchema().WithAutoID(true);
    EXPECT_TRUE(schema.AutoID());

    auto schema2 = schema.WithAutoID(false);
    EXPECT_FALSE(schema2.AutoID());
}

TEST_F(FieldSchemaTest, PartitionKey) {
    milvus::FieldSchema schema;
    EXPECT_FALSE(schema.IsPartitionKey());

    schema.SetPartitionKey(true);
    EXPECT_TRUE(schema.IsPartitionKey());

    auto& ref = schema.WithPartitionKey(false);
    EXPECT_FALSE(ref.IsPartitionKey());
}

TEST_F(FieldSchemaTest, ClusteringKey) {
    milvus::FieldSchema schema;
    EXPECT_FALSE(schema.IsClusteringKey());

    schema.SetClusteringKey(true);
    EXPECT_TRUE(schema.IsClusteringKey());

    auto& ref = schema.WithClusteringKey(false);
    EXPECT_FALSE(ref.IsClusteringKey());
}

TEST_F(FieldSchemaTest, AddTypeParam) {
    milvus::FieldSchema schema;
    schema.AddTypeParam("key1", "val1");
    EXPECT_EQ("val1", schema.TypeParams().at("key1"));
}

TEST_F(FieldSchemaTest, SetAndWithMaxLength) {
    milvus::FieldSchema schema("name", milvus::DataType::VARCHAR, "");
    schema.SetMaxLength(500);
    EXPECT_EQ(500, schema.MaxLength());

    auto& ref = schema.WithMaxLength(1000);
    EXPECT_EQ(1000, ref.MaxLength());
}

TEST_F(FieldSchemaTest, MaxCapacitySetAndWith) {
    milvus::FieldSchema schema("arr", milvus::DataType::ARRAY, "");
    EXPECT_EQ(0, schema.MaxCapacity());

    schema.SetMaxCapacity(200);
    EXPECT_EQ(200, schema.MaxCapacity());

    auto& ref = schema.WithMaxCapacity(300);
    EXPECT_EQ(300, ref.MaxCapacity());
}

TEST_F(FieldSchemaTest, EnableAnalyzerAndMatch) {
    milvus::FieldSchema schema("text", milvus::DataType::VARCHAR, "");
    EXPECT_FALSE(schema.IsEnableAnalyzer());
    EXPECT_FALSE(schema.IsEnableMatch());

    schema.EnableAnalyzer(true);
    EXPECT_TRUE(schema.IsEnableAnalyzer());

    schema.EnableMatch(true);
    EXPECT_TRUE(schema.IsEnableMatch());

    schema.EnableAnalyzer(false);
    EXPECT_FALSE(schema.IsEnableAnalyzer());

    schema.EnableMatch(false);
    EXPECT_FALSE(schema.IsEnableMatch());
}

TEST_F(FieldSchemaTest, AnalyzerParams) {
    milvus::FieldSchema schema("text", milvus::DataType::VARCHAR, "");
    nlohmann::json params = {{"type", "standard"}};

    schema.SetAnalyzerParams(params);
    EXPECT_EQ(schema.AnalyzerParams(), params);

    nlohmann::json params2 = {{"type", "chinese"}};
    auto& ref = schema.WithAnalyzerParams(params2);
    EXPECT_EQ(ref.AnalyzerParams(), params2);
}

TEST_F(FieldSchemaTest, MultiAnalyzerParams) {
    milvus::FieldSchema schema("text", milvus::DataType::VARCHAR, "");
    nlohmann::json params = {{"en", {{"type", "standard"}}}, {"zh", {{"type", "chinese"}}}};

    schema.SetMultiAnalyzerParams(params);
    EXPECT_EQ(schema.MultiAnalyzerParams(), params);

    nlohmann::json params2 = {{"de", {{"type", "german"}}}};
    auto& ref = schema.WithMultiAnalyzerParams(params2);
    EXPECT_EQ(ref.MultiAnalyzerParams(), params2);
}

TEST_F(FieldSchemaTest, Nullable) {
    milvus::FieldSchema schema;
    EXPECT_FALSE(schema.IsNullable());

    schema.SetNullable(true);
    EXPECT_TRUE(schema.IsNullable());

    auto& ref = schema.WithNullable(false);
    EXPECT_FALSE(ref.IsNullable());
}

TEST_F(FieldSchemaTest, DefaultValue) {
    milvus::FieldSchema schema;
    EXPECT_TRUE(schema.DefaultValue().is_null());

    nlohmann::json val = 42;
    schema.SetDefaultValue(val);
    EXPECT_EQ(schema.DefaultValue(), 42);

    nlohmann::json val2 = "hello";
    auto& ref = schema.WithDefaultValue(val2);
    EXPECT_EQ(ref.DefaultValue(), "hello");
}