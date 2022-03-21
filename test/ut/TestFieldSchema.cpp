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
    EXPECT_TRUE(type_params.find(milvus::FieldDim()) != type_params.end());
    EXPECT_EQ("256", type_params.at(milvus::FieldDim()));
}

TEST_F(FieldSchemaTest, TestWithDimention) {
    EXPECT_EQ("1024", milvus::FieldSchema("vectors", milvus::DataType::FLOAT_VECTOR, "")
                          .WithDimension(1024)
                          .TypeParams()
                          .at(milvus::FieldDim()));
}
