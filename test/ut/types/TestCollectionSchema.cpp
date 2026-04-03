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

#include "milvus/types/CollectionSchema.h"

class CollectionSchemaTest : public ::testing::Test {};

TEST_F(CollectionSchemaTest, GeneralTesting) {
    milvus::CollectionSchema schema;
    schema.SetName("test");
    schema.SetDescription("test");

    milvus::FieldSchema id_field{"foo", milvus::DataType::INT64, "foo"};
    EXPECT_TRUE(schema.AddField(id_field));
    EXPECT_FALSE(schema.AddField(id_field));

    EXPECT_TRUE(schema.AddField(milvus::FieldSchema("bar", milvus::DataType::FLOAT_VECTOR, "bar")));
    EXPECT_FALSE(schema.AddField(milvus::FieldSchema("bar", milvus::DataType::FLOAT_VECTOR, "bar")));

    EXPECT_EQ(schema.ShardsNum(), 1);
    EXPECT_EQ(schema.Fields().size(), 2);
    EXPECT_EQ(schema.AnnsFieldNames().size(), 1);
    EXPECT_EQ(*schema.AnnsFieldNames().begin(), "bar");
}

TEST_F(CollectionSchemaTest, Description) {
    milvus::CollectionSchema schema;
    schema.SetDescription("my description");
    EXPECT_EQ(schema.Description(), "my description");

    milvus::CollectionSchema schema2("name", "desc2", 2, false);
    EXPECT_EQ(schema2.Description(), "desc2");
}

TEST_F(CollectionSchemaTest, SetShardsNum) {
    milvus::CollectionSchema schema;
    EXPECT_EQ(schema.ShardsNum(), 1);
    schema.SetShardsNum(4);
    EXPECT_EQ(schema.ShardsNum(), 4);
}

TEST_F(CollectionSchemaTest, EnableDynamicField) {
    milvus::CollectionSchema schema;
    EXPECT_TRUE(schema.EnableDynamicField());

    schema.SetEnableDynamicField(false);
    EXPECT_FALSE(schema.EnableDynamicField());

    schema.SetEnableDynamicField(true);
    EXPECT_TRUE(schema.EnableDynamicField());

    milvus::CollectionSchema schema2("name", "desc", 1, false);
    EXPECT_FALSE(schema2.EnableDynamicField());
}

TEST_F(CollectionSchemaTest, StructFields) {
    milvus::CollectionSchema schema;
    EXPECT_TRUE(schema.StructFields().empty());

    milvus::StructFieldSchema struct_field("struct1", "desc");
    struct_field.AddField(milvus::FieldSchema("sub1", milvus::DataType::INT32, ""));
    EXPECT_TRUE(schema.AddStructField(struct_field));
    EXPECT_EQ(schema.StructFields().size(), 1);
    EXPECT_EQ(schema.StructFields().at(0).Name(), "struct1");
}

TEST_F(CollectionSchemaTest, PrimaryFieldName) {
    milvus::CollectionSchema schema;
    EXPECT_EQ(schema.PrimaryFieldName(), "");

    milvus::FieldSchema pk("id", milvus::DataType::INT64, "primary key", true, true);
    schema.AddField(pk);
    EXPECT_EQ(schema.PrimaryFieldName(), "id");
}

TEST_F(CollectionSchemaTest, Functions) {
    milvus::CollectionSchema schema;
    EXPECT_TRUE(schema.Functions().empty());

    auto func = std::make_shared<milvus::Function>("bm25_func", milvus::FunctionType::BM25, "bm25 function");
    schema.AddFunction(func);
    EXPECT_EQ(schema.Functions().size(), 1);
    EXPECT_EQ(schema.Functions().at(0)->Name(), "bm25_func");
}
