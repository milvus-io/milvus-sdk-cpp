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

class MilvusServerTestIndex : public MilvusServerTest {
 protected:
    std::string collection_name;

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        collection_name = milvus::test::RanName("IdxTest_");

        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
        schema->AddField(milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name").WithMaxLength(64));
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

TEST_F(MilvusServerTestIndex, CreateDescribeListDrop) {
    // create index
    milvus::IndexDesc index_desc("vec", "my_index", milvus::IndexType::IVF_FLAT, milvus::MetricType::L2);
    index_desc.AddExtraParam("nlist", "128");
    auto status = client_->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
    std::cout << "index created" << std::endl;
    milvus::test::ExpectStatusOK(status);

    auto check_description = [](const milvus::DescribeIndexResponse& desc_resp) {
        EXPECT_EQ(desc_resp.Descs().size(), 1);
        EXPECT_EQ(desc_resp.Descs()[0].IndexName(), "my_index");
        EXPECT_EQ(desc_resp.Descs()[0].IndexType(), milvus::IndexType::IVF_FLAT);
        EXPECT_EQ(desc_resp.Descs()[0].MetricType(), milvus::MetricType::L2);
        auto extra_params = desc_resp.Descs()[0].ExtraParams();
        EXPECT_EQ(extra_params.size(), 1);
        EXPECT_EQ(extra_params["nlist"], "128");
    };

    // describe index with field name
    milvus::DescribeIndexResponse desc_resp;
    status = client_->DescribeIndex(
        milvus::DescribeIndexRequest().WithCollectionName(collection_name).WithFieldName("vec"), desc_resp);
    milvus::test::ExpectStatusOK(status);
    check_description(desc_resp);

    // describe index with index name
    status = client_->DescribeIndex(
        milvus::DescribeIndexRequest().WithCollectionName(collection_name).WithIndexName("my_index"), desc_resp);
    milvus::test::ExpectStatusOK(status);
    check_description(desc_resp);

    // list indexes
    milvus::ListIndexesResponse list_resp;
    status = client_->ListIndexes(milvus::ListIndexesRequest().WithCollectionName(collection_name), list_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(list_resp.IndexNames().size(), 1);
    EXPECT_EQ(list_resp.IndexNames()[0], "my_index");

    // drop index
    status =
        client_->DropIndex(milvus::DropIndexRequest().WithCollectionName(collection_name).WithIndexName("my_index"));
    milvus::test::ExpectStatusOK(status);
    std::cout << "index dropped" << std::endl;

    // verify dropped
    milvus::ListIndexesResponse list_resp2;
    status = client_->ListIndexes(milvus::ListIndexesRequest().WithCollectionName(collection_name), list_resp2);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(list_resp2.IndexNames().size(), 0);
}

TEST_F(MilvusServerTestIndex, CreateHNSWIndex) {
    milvus::IndexDesc index_desc("vec", "hnsw_index", milvus::IndexType::HNSW, milvus::MetricType::L2);
    index_desc.AddExtraParam("M", "16");
    index_desc.AddExtraParam("efConstruction", "200");
    auto status = client_->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
    milvus::test::ExpectStatusOK(status);

    milvus::DescribeIndexResponse desc_resp;
    status = client_->DescribeIndex(
        milvus::DescribeIndexRequest().WithCollectionName(collection_name).WithFieldName("vec"), desc_resp);
    milvus::test::ExpectStatusOK(status);
}

TEST_F(MilvusServerTestIndex, AlterAndDropIndexProperties) {
    // create index first
    milvus::IndexDesc index_desc("vec", "test_index", milvus::IndexType::HNSW, milvus::MetricType::L2);
    index_desc.AddExtraParam("M", "16");
    index_desc.AddExtraParam("efConstruction", "200");
    auto status = client_->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
    milvus::test::ExpectStatusOK(status);

    // alter index properties
    status = client_->AlterIndexProperties(milvus::AlterIndexPropertiesRequest()
                                               .WithCollectionName(collection_name)
                                               .WithIndexName("test_index")
                                               .AddProperty("mmap.enabled", "true"));
    milvus::test::ExpectStatusOK(status);

    // verify by describing the index
    milvus::DescribeIndexResponse desc_resp;
    status = client_->DescribeIndex(
        milvus::DescribeIndexRequest().WithCollectionName(collection_name).WithIndexName("test_index"), desc_resp);
    milvus::test::ExpectStatusOK(status);

    EXPECT_EQ(desc_resp.Descs().size(), 1);
    auto params = desc_resp.Descs()[0].ExtraParams();
    EXPECT_EQ(params.size(), 3);  // M, efConstruction, mmap.enabled
    EXPECT_EQ(params["M"], "16");
    EXPECT_EQ(params["efConstruction"], "200");
    EXPECT_EQ(params["mmap.enabled"], "true");

    // drop index properties
    status = client_->DropIndexProperties(milvus::DropIndexPropertiesRequest()
                                              .WithCollectionName(collection_name)
                                              .WithIndexName("test_index")
                                              .AddPropertyKey("mmap.enabled"));
    milvus::test::ExpectStatusOK(status);

    // verify again by describing the index
    status = client_->DescribeIndex(
        milvus::DescribeIndexRequest().WithCollectionName(collection_name).WithIndexName("test_index"), desc_resp);
    milvus::test::ExpectStatusOK(status);

    EXPECT_EQ(desc_resp.Descs().size(), 1);
    params = desc_resp.Descs()[0].ExtraParams();
    EXPECT_EQ(params.size(), 2);  // M, efConstruction
    EXPECT_TRUE(params.find("mmap.enabled") == params.end());
}
