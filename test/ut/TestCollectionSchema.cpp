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

    EXPECT_EQ(schema.ShardsNum(), 2);
    EXPECT_EQ(schema.Fields().size(), 2);
    EXPECT_EQ(schema.AnnsFieldNames().size(), 1);
    EXPECT_EQ(*schema.AnnsFieldNames().begin(), "bar");
}
