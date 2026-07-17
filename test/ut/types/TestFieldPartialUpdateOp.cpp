// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include "milvus/types/FieldPartialUpdateOp.h"

class FieldPartialUpdateOpTest : public ::testing::Test {};

TEST_F(FieldPartialUpdateOpTest, ConstructorsAndAccessors) {
    milvus::FieldPartialUpdateOp field_op;
    EXPECT_TRUE(field_op.FieldName().empty());
    EXPECT_EQ(field_op.GetOpType(), milvus::FieldPartialUpdateOp::OpType::REPLACE);

    field_op.SetFieldName("tags");
    field_op.SetOpType(milvus::FieldPartialUpdateOp::OpType::ARRAY_APPEND);
    EXPECT_EQ(field_op.FieldName(), "tags");
    EXPECT_EQ(field_op.GetOpType(), milvus::FieldPartialUpdateOp::OpType::ARRAY_APPEND);

    auto& ref = field_op.WithFieldName("labels").WithOpType(milvus::FieldPartialUpdateOp::OpType::ARRAY_REMOVE);
    EXPECT_EQ(&ref, &field_op);
    EXPECT_EQ(field_op.FieldName(), "labels");
    EXPECT_EQ(field_op.GetOpType(), milvus::FieldPartialUpdateOp::OpType::ARRAY_REMOVE);

    milvus::FieldPartialUpdateOp constructed("scores", milvus::FieldPartialUpdateOp::OpType::ARRAY_APPEND);
    EXPECT_EQ(constructed.FieldName(), "scores");
    EXPECT_EQ(constructed.GetOpType(), milvus::FieldPartialUpdateOp::OpType::ARRAY_APPEND);
}
