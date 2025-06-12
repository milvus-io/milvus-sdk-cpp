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

#include "milvus/types/IndexDesc.h"

class IndexDescTest : public ::testing::Test {};

TEST_F(IndexDescTest, GeneralTesting) {
    milvus::IndexDesc index_desc;

    index_desc.SetFieldName("field_name");
    EXPECT_EQ(index_desc.FieldName(), "field_name");

    index_desc.SetIndexName("index_name");
    EXPECT_EQ(index_desc.IndexName(), "index_name");

    index_desc.SetIndexId(1);
    EXPECT_EQ(index_desc.IndexId(), 1);

    index_desc.SetIndexType(milvus::IndexType::IVF_FLAT);
    EXPECT_EQ(index_desc.IndexType(), milvus::IndexType::IVF_FLAT);

    index_desc.SetIndexedRows(5);
    EXPECT_EQ(index_desc.IndexedRows(), 5);

    index_desc.SetPendingRows(6);
    EXPECT_EQ(index_desc.PendingRows(), 6);

    index_desc.SetTotalRows(7);
    EXPECT_EQ(index_desc.TotalRows(), 7);

    index_desc.SetStateCode(milvus::IndexStateCode::IN_PROGRESS);
    EXPECT_EQ(index_desc.StateCode(), milvus::IndexStateCode::IN_PROGRESS);

    index_desc.SetFailReason("hello failed");
    EXPECT_EQ(index_desc.FailReason(), "hello failed");
}
