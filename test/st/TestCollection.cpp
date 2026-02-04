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

#include <algorithm>

#include "MilvusServerTest.h"

using milvus::test::MilvusServerTestWithParam;
using MilvusServerTestCollection = MilvusServerTestWithParam<bool>;

bool
ContainsCollection(const std::vector<milvus::CollectionInfo>& collection_infos, const std::string& name) {
    auto it = std::find_if(collection_infos.begin(), collection_infos.end(),
                           [&name](const milvus::CollectionInfo& info) { return info.Name() == name; });
    return (it != collection_infos.end());
}

TEST_P(MilvusServerTestCollection, CreateAndDeleteCollection) {
    auto using_string_primary_key = GetParam();

    std::string collection_name = milvus::test::RanName("Foo_");
    auto collection_schema = std::make_shared<milvus::CollectionSchema>(collection_name);
    if (using_string_primary_key) {
        collection_schema->AddField(
            // string as primary key, no auto-id
            milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name", true, false).WithMaxLength(64));
    } else {
        collection_schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        collection_schema->AddField(
            milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name").WithMaxLength(64));
    }
    collection_schema->AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
    collection_schema->AddField(
        milvus::FieldSchema("face", milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(1024));

    auto status = client_->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    EXPECT_EQ(status.Message(), "OK");
    milvus::test::ExpectStatusOK(status);

    // create index needed after 2.2.0
    milvus::IndexDesc index_desc("face", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    status = client_->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
    milvus::test::ExpectStatusOK(status);

    // test for https://github.com/milvus-io/milvus-sdk-cpp/issues/188
    milvus::ListCollectionsResponse list_resp;
    status = client_->ListCollections(milvus::ListCollectionsRequest(), list_resp);
    milvus::test::ExpectStatusOK(status);
    auto& collection_infos = list_resp.CollectionInfos();
    EXPECT_GE(collection_infos.size(), 1);
    EXPECT_TRUE(ContainsCollection(collection_infos, collection_name));

    // test for https://github.com/milvus-io/milvus-sdk-cpp/issues/246
    milvus::ListPartitionsResponse lp_resp;
    status = client_->ListPartitions(
        milvus::ListPartitionsRequest().WithCollectionName(collection_name), lp_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_GE(lp_resp.PartitionsNames().size(), 1);

    // the collection is not loaded, set only_show_loaded = true, the collection is not in the list
    milvus::ListCollectionsResponse list_resp2;
    status = client_->ListCollections(milvus::ListCollectionsRequest().WithOnlyShowLoaded(true), list_resp2);
    milvus::test::ExpectStatusOK(status);
    auto& loaded_infos = list_resp2.CollectionInfos();
    EXPECT_FALSE(ContainsCollection(loaded_infos, collection_name));

    // load the collection
    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    // the collection is loaded, set only_show_loaded = true, the collection is in the list
    milvus::ListCollectionsResponse list_resp3;
    status = client_->ListCollections(milvus::ListCollectionsRequest(), list_resp3);
    milvus::test::ExpectStatusOK(status);
    auto& all_infos = list_resp3.CollectionInfos();
    EXPECT_TRUE(ContainsCollection(all_infos, collection_name));

    status = client_->RenameCollection(
        milvus::RenameCollectionRequest().WithCollectionName(collection_name).WithNewCollectionName("Bar"));
    milvus::test::ExpectStatusOK(status);

    // the collection is dropped, not in the list of ListCollections
    status = client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName("Bar"));
    milvus::test::ExpectStatusOK(status);
    milvus::ListCollectionsResponse list_resp4;
    status = client_->ListCollections(milvus::ListCollectionsRequest(), list_resp4);
    milvus::test::ExpectStatusOK(status);
    auto& final_infos = list_resp4.CollectionInfos();
    EXPECT_FALSE(ContainsCollection(final_infos, collection_name));
}

INSTANTIATE_TEST_SUITE_P(SystemTest, MilvusServerTestCollection, ::testing::Values(false, true));
