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

class DatabaseDescTest : public ::testing::Test {};

TEST_F(DatabaseDescTest, NameGetterSetter) {
    milvus::DatabaseDesc desc;
    desc.SetName("test_db");
    EXPECT_EQ(desc.Name(), "test_db");
}

TEST_F(DatabaseDescTest, IDGetterSetter) {
    milvus::DatabaseDesc desc;
    desc.SetID(42);
    EXPECT_EQ(desc.ID(), 42);
}

TEST_F(DatabaseDescTest, PropertiesGetterSetter) {
    milvus::DatabaseDesc desc;

    std::unordered_map<std::string, std::string> props;
    props["key1"] = "value1";
    props["key2"] = "value2";
    desc.SetProperties(props);

    EXPECT_EQ(desc.Properties().size(), 2);
    EXPECT_EQ(desc.Properties().at("key1"), "value1");
    EXPECT_EQ(desc.Properties().at("key2"), "value2");
}

TEST_F(DatabaseDescTest, PropertiesMoveGetterSetter) {
    milvus::DatabaseDesc desc;

    std::unordered_map<std::string, std::string> props;
    props["k"] = "v";
    desc.SetProperties(std::move(props));

    EXPECT_EQ(desc.Properties().size(), 1);
    EXPECT_EQ(desc.Properties().at("k"), "v");
}

TEST_F(DatabaseDescTest, CreatedTimeGetterSetter) {
    milvus::DatabaseDesc desc;
    EXPECT_EQ(desc.CreatedTime(), 0u);

    desc.SetCreatedTime(1234567890);
    EXPECT_EQ(desc.CreatedTime(), 1234567890u);
}
