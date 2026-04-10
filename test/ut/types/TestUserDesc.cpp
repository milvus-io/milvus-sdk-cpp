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

class UserDescTest : public ::testing::Test {};

TEST_F(UserDescTest, DefaultConstructor) {
    milvus::UserDesc desc;
    EXPECT_TRUE(desc.Name().empty());
    EXPECT_TRUE(desc.Roles().empty());
}

TEST_F(UserDescTest, ParameterizedConstructor) {
    std::vector<std::string> roles = {"admin", "reader"};
    milvus::UserDesc desc("alice", std::move(roles));
    EXPECT_EQ(desc.Name(), "alice");
    EXPECT_EQ(desc.Roles().size(), 2);
    EXPECT_EQ(desc.Roles()[0], "admin");
    EXPECT_EQ(desc.Roles()[1], "reader");
}

TEST_F(UserDescTest, SetName) {
    milvus::UserDesc desc;
    desc.SetName("bob");
    EXPECT_EQ(desc.Name(), "bob");
}

TEST_F(UserDescTest, AddRole) {
    milvus::UserDesc desc;
    desc.AddRole("writer");
    desc.AddRole("viewer");
    EXPECT_EQ(desc.Roles().size(), 2);
    EXPECT_EQ(desc.Roles()[0], "writer");
    EXPECT_EQ(desc.Roles()[1], "viewer");
}
