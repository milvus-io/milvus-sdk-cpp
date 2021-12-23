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

#include "types/FieldSchema.h"

class FieldSchemaTest : public ::testing::Test {};

TEST_F(FieldSchemaTest, GeneralTesting) {
    std::string name = "f0";
    std::string desc = "desc";
    milvus::DataType dt = milvus::DataType::DOUBLE;
    bool is_primary_key = true;
    bool auto_id = false;

    milvus::FieldSchema schema(name, dt, desc, is_primary_key, auto_id);
    EXPECT_EQ(name, schema.Name());
    EXPECT_EQ(desc, schema.Description());
    EXPECT_EQ(dt, schema.FieldDataType());
    EXPECT_EQ(is_primary_key, schema.IsPrimaryKey());
    EXPECT_EQ(auto_id, schema.AutoID());
}
