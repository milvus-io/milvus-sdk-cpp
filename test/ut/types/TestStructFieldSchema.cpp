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

#include "milvus/MilvusClientV2.h"

class StructFieldSchemaTest : public ::testing::Test {};

TEST_F(StructFieldSchemaTest, DefaultConstructor) {
    milvus::StructFieldSchema schema;
    EXPECT_TRUE(schema.Name().empty());
    EXPECT_TRUE(schema.Description().empty());
    EXPECT_EQ(schema.MaxCapacity(), 0);
    EXPECT_TRUE(schema.Fields().empty());
}

TEST_F(StructFieldSchemaTest, ParameterizedConstructor) {
    milvus::StructFieldSchema schema("struct_field", "a description");
    EXPECT_EQ(schema.Name(), "struct_field");
    EXPECT_EQ(schema.Description(), "a description");
}

TEST_F(StructFieldSchemaTest, WithName) {
    milvus::StructFieldSchema schema;
    auto& ref = schema.WithName("my_struct");
    EXPECT_EQ(schema.Name(), "my_struct");
    EXPECT_EQ(&ref, &schema);
}

TEST_F(StructFieldSchemaTest, SetName) {
    milvus::StructFieldSchema schema;
    schema.SetName("field_name");
    EXPECT_EQ(schema.Name(), "field_name");
}

TEST_F(StructFieldSchemaTest, WithDescription) {
    milvus::StructFieldSchema schema;
    auto& ref = schema.WithDescription("desc");
    EXPECT_EQ(schema.Description(), "desc");
    EXPECT_EQ(&ref, &schema);
}

TEST_F(StructFieldSchemaTest, SetDescription) {
    milvus::StructFieldSchema schema;
    schema.SetDescription("my desc");
    EXPECT_EQ(schema.Description(), "my desc");
}

TEST_F(StructFieldSchemaTest, WithMaxCapacity) {
    milvus::StructFieldSchema schema;
    auto& ref = schema.WithMaxCapacity(100);
    EXPECT_EQ(schema.MaxCapacity(), 100);
    EXPECT_EQ(&ref, &schema);
}

TEST_F(StructFieldSchemaTest, SetMaxCapacity) {
    milvus::StructFieldSchema schema;
    schema.SetMaxCapacity(200);
    EXPECT_EQ(schema.MaxCapacity(), 200);
}

TEST_F(StructFieldSchemaTest, AddFieldLvalue) {
    milvus::StructFieldSchema schema;
    milvus::FieldSchema field("sub_field", milvus::DataType::INT64);
    auto& ref = schema.AddField(field);
    EXPECT_EQ(schema.Fields().size(), 1);
    EXPECT_EQ(schema.Fields()[0].Name(), "sub_field");
    EXPECT_EQ(&ref, &schema);
}

TEST_F(StructFieldSchemaTest, AddFieldRvalue) {
    milvus::StructFieldSchema schema;
    auto& ref = schema.AddField(milvus::FieldSchema("sub_field_2", milvus::DataType::FLOAT));
    EXPECT_EQ(schema.Fields().size(), 1);
    EXPECT_EQ(schema.Fields()[0].Name(), "sub_field_2");
    EXPECT_EQ(&ref, &schema);
}

TEST_F(StructFieldSchemaTest, ChainingMethods) {
    milvus::StructFieldSchema schema;
    schema.WithName("s1")
        .WithDescription("desc")
        .WithMaxCapacity(50)
        .AddField(milvus::FieldSchema("f1", milvus::DataType::VARCHAR))
        .AddField(milvus::FieldSchema("f2", milvus::DataType::INT32));
    EXPECT_EQ(schema.Name(), "s1");
    EXPECT_EQ(schema.Description(), "desc");
    EXPECT_EQ(schema.MaxCapacity(), 50);
    EXPECT_EQ(schema.Fields().size(), 2);
}
