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

class ListResourceGroupsResponseTest : public ::testing::Test {};

TEST_F(ListResourceGroupsResponseTest, SetterAndGetter) {
    milvus::ListResourceGroupsResponse resp;
    std::vector<std::string> names{"rg1", "rg2"};
    resp.SetGroupNames(std::move(names));
    EXPECT_EQ(resp.GroupNames().size(), 2);
}

class DescribeResourceGroupResponseTest : public ::testing::Test {};

TEST_F(DescribeResourceGroupResponseTest, SetterAndGetter) {
    milvus::DescribeResourceGroupResponse resp;
    milvus::ResourceGroupDesc desc;
    resp.SetDesc(std::move(desc));
    (void)resp.Desc();
}
