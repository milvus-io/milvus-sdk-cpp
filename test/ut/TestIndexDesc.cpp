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

#include "types/IndexDesc.h"

class IndexDescTest : public ::testing::Test {};

TEST_F(IndexDescTest, GeneralTesting) {
    // std::string field_name = "f0";
    // std::string index_name = "IVF";
    // int64_t index_id = 99;
    // std::unordered_map<std::string, std::string> params;
    // params["nlist"] = "10";

    // milvus::IndexDesc desc(field_name, index_name, index_id, params);
    // EXPECT_EQ(field_name, desc.FieldName());
    // EXPECT_EQ(index_name, desc.IndexId());
    // EXPECT_EQ(index_id, desc.IndexId());
    // EXPECT_EQ("10", desc.Params().["nlist"]);
}
