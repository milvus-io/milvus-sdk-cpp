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

class DescribeIndexResponseTest : public ::testing::Test {};

TEST_F(DescribeIndexResponseTest, SetterAndGetter) {
    milvus::DescribeIndexResponse resp;
    std::vector<milvus::IndexDesc> descs;
    milvus::IndexDesc idx;
    descs.push_back(idx);
    resp.SetDescs(std::move(descs));
    EXPECT_EQ(resp.Descs().size(), 1);
}

class ListIndexesResponseTest : public ::testing::Test {};

TEST_F(ListIndexesResponseTest, SetterAndGetter) {
    milvus::ListIndexesResponse resp;
    std::vector<std::string> names{"idx1", "idx2"};
    resp.SetIndexNames(std::move(names));
    EXPECT_EQ(resp.IndexNames().size(), 2);
    EXPECT_EQ(resp.IndexNames()[0], "idx1");
}
