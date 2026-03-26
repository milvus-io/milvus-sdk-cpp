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

using milvus::test::MilvusServerTest;

class MilvusServerTestAlias : public MilvusServerTest {
 protected:
    std::string collection_name;

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        collection_name = milvus::test::RanName("AliasTest_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        schema->AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(4));

        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);
    }

    void
    TearDown() override {
        client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
        MilvusServerTest::TearDown();
    }
};

TEST_F(MilvusServerTestAlias, CreateDescribeListDrop) {
    std::string alias_name = milvus::test::RanName("alias_");

    // create alias
    auto status =
        client_->CreateAlias(milvus::CreateAliasRequest().WithCollectionName(collection_name).WithAlias(alias_name));
    milvus::test::ExpectStatusOK(status);

    // describe alias
    milvus::DescribeAliasResponse desc_resp;
    status = client_->DescribeAlias(milvus::DescribeAliasRequest().WithAlias(alias_name), desc_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(collection_name, desc_resp.Desc().CollectionName());
    EXPECT_EQ(alias_name, desc_resp.Desc().Name());

    // list aliases for collection
    milvus::ListAliasesResponse list_resp;
    status = client_->ListAliases(milvus::ListAliasesRequest().WithCollectionName(collection_name), list_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(collection_name, list_resp.CollectionName());
    EXPECT_EQ(1, list_resp.Aliases().size());
    EXPECT_EQ(alias_name, list_resp.Aliases().front());

    // drop alias
    status = client_->DropAlias(milvus::DropAliasRequest().WithAlias(alias_name));
    milvus::test::ExpectStatusOK(status);
}

TEST_F(MilvusServerTestAlias, AlterAlias) {
    std::string alias_name = milvus::test::RanName("alias_");
    std::string collection_name2 = milvus::test::RanName("AliasTest2_");

    // create second collection
    auto schema2 = std::make_shared<milvus::CollectionSchema>(collection_name2);
    schema2->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
    schema2->AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(4));
    auto status = client_->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name2).WithCollectionSchema(schema2));
    milvus::test::ExpectStatusOK(status);

    // create alias pointing to first collection
    status =
        client_->CreateAlias(milvus::CreateAliasRequest().WithCollectionName(collection_name).WithAlias(alias_name));
    milvus::test::ExpectStatusOK(status);

    // alter alias to point to second collection
    status =
        client_->AlterAlias(milvus::AlterAliasRequest().WithCollectionName(collection_name2).WithAlias(alias_name));
    milvus::test::ExpectStatusOK(status);

    // describe alias
    milvus::DescribeAliasResponse desc_resp;
    status = client_->DescribeAlias(milvus::DescribeAliasRequest().WithAlias(alias_name), desc_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(collection_name2, desc_resp.Desc().CollectionName());
    EXPECT_EQ(alias_name, desc_resp.Desc().Name());

    // cleanup
    client_->DropAlias(milvus::DropAliasRequest().WithAlias(alias_name));
    client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name2));
}
