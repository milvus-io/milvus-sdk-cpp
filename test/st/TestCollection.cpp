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

#include "MilvusServerTest.h"

using milvus::test::MilvusServerTestWithParam;

using MilvusServerTestCollection = MilvusServerTestWithParam<bool>;

TEST_P(MilvusServerTestCollection, CreateAndDeleteCollection) {
    auto using_string_primary_key = GetParam();

    milvus::CollectionSchema collection_schema("Foo");
    if (using_string_primary_key) {
        collection_schema.AddField(
            // string as primary key, no auto-id
            milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name", true, false).WithMaxLength(64));
    } else {
        collection_schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        collection_schema.AddField(milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name").WithMaxLength(64));
    }
    collection_schema.AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
    collection_schema.AddField(
        milvus::FieldSchema("face", milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(1024));

    auto status = client_->CreateCollection(collection_schema);
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_TRUE(status.IsOk());

    // create index needed after 2.2.0
    milvus::IndexDesc index_desc("face", "", milvus::IndexType::FLAT, milvus::MetricType::L2, 0);
    status = client_->CreateIndex("Foo", index_desc);
    EXPECT_TRUE(status.IsOk());

    // test for https://github.com/milvus-io/milvus-sdk-cpp/issues/188
    std::vector<std::string> names;
    std::vector<milvus::CollectionInfo> collection_infos;
    status = client_->ShowCollections(names, collection_infos);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(collection_infos.size(), 1);
    EXPECT_EQ(collection_infos.front().MemoryPercentage(), 0);
    EXPECT_EQ(collection_infos.front().Name(), "Foo");

    // test for https://github.com/milvus-io/milvus-sdk-cpp/issues/246
    milvus::PartitionsInfo partitionsInfo{};
    status = client_->ShowPartitions("Foo", std::vector<std::string>{}, partitionsInfo);
    EXPECT_TRUE(status.IsOk());

    names.emplace_back("Foo");
    collection_infos.clear();
    status = client_->LoadCollection("Foo");
    EXPECT_TRUE(status.IsOk());

    status = client_->ShowCollections(names, collection_infos);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(collection_infos.size(), 1);
    EXPECT_EQ(collection_infos.front().MemoryPercentage(), 100);

    status = client_->RenameCollection("Foo", "Bar");
    EXPECT_TRUE(status.IsOk());

    status = client_->DropCollection("Bar");
    EXPECT_TRUE(status.IsOk());
}

INSTANTIATE_TEST_SUITE_P(SystemTest, MilvusServerTestCollection, ::testing::Values(false, true));
