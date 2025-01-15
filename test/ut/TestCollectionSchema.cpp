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

    EXPECT_TRUE(schema.AddField(milvus::FieldSchema("fp16", milvus::DataType::FLOAT16_VECTOR, "fp16")));
    EXPECT_FALSE(schema.AddField(milvus::FieldSchema("fp16", milvus::DataType::FLOAT16_VECTOR, "fp16")));

    EXPECT_TRUE(schema.AddField(milvus::FieldSchema("bf16", milvus::DataType::BFLOAT16_VECTOR, "bf16")));
    EXPECT_FALSE(schema.AddField(milvus::FieldSchema("bf16", milvus::DataType::BFLOAT16_VECTOR, "bf16")));

    EXPECT_EQ(schema.ShardsNum(), 2);
    EXPECT_EQ(schema.Fields().size(), 4);
    auto anns_field_names = schema.AnnsFieldNames();
    EXPECT_EQ(anns_field_names.size(), 3);
    EXPECT_TRUE(anns_field_names.find("bar") != anns_field_names.end());
    EXPECT_TRUE(anns_field_names.find("fp16") != anns_field_names.end());
    EXPECT_TRUE(anns_field_names.find("bf16") != anns_field_names.end());
}
