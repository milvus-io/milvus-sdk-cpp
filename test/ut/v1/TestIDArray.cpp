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

#include "milvus/types/IDArray.h"

class IDArrayTest : public ::testing::Test {};

TEST_F(IDArrayTest, GeneralTesting) {
    {
        std::vector<int64_t> ids = {1, 2, 3};
        milvus::IDArray id_array_foo(ids);
        EXPECT_TRUE(id_array_foo.IsIntegerID());
        EXPECT_EQ(id_array_foo.IntIDArray().size(), 3);
        EXPECT_TRUE(id_array_foo.StrIDArray().empty());

        milvus::IDArray id_array_bar(std::move(ids));
        EXPECT_TRUE(id_array_bar.IsIntegerID());
        EXPECT_EQ(id_array_bar.IntIDArray().size(), 3);
        EXPECT_TRUE(id_array_bar.StrIDArray().empty());
    }

    {
        std::vector<std::string> ids = {"a", "b", "c"};
        milvus::IDArray id_array_foo(ids);
        EXPECT_FALSE(id_array_foo.IsIntegerID());
        EXPECT_EQ(id_array_foo.StrIDArray().size(), 3);
        EXPECT_TRUE(id_array_foo.IntIDArray().empty());

        milvus::IDArray id_array_bar(std::move(ids));
        EXPECT_FALSE(id_array_bar.IsIntegerID());
        EXPECT_EQ(id_array_bar.StrIDArray().size(), 3);
        EXPECT_TRUE(id_array_bar.IntIDArray().empty());
    }
}
