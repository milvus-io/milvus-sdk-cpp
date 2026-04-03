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
        collection_schema->AddField(milvus::FieldSchema("name", milvus::DataType::VARCHAR, "name").WithMaxLength(64));
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
    status = client_->ListPartitions(milvus::ListPartitionsRequest().WithCollectionName(collection_name), lp_resp);
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

class MilvusServerTestCollectionOps : public milvus::test::MilvusServerTest {
 protected:
    std::string collection_name;

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        collection_name = milvus::test::RanName("CollOps_");

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

TEST_F(MilvusServerTestCollectionOps, HasAndDescribeCollection) {
    // has collection
    milvus::HasCollectionResponse has_resp;
    auto status = client_->HasCollection(milvus::HasCollectionRequest().WithCollectionName(collection_name), has_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_TRUE(has_resp.Has());

    // describe collection
    milvus::DescribeCollectionResponse desc_resp;
    status =
        client_->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name), desc_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(collection_name, desc_resp.Desc().CollectionName());
    auto schema = desc_resp.Desc().Schema();
    EXPECT_EQ(3, schema.Fields().size());
    EXPECT_EQ(milvus::DataType::INT64, schema.Fields().at(0).FieldDataType());
    EXPECT_EQ(milvus::DataType::VARCHAR, schema.Fields().at(1).FieldDataType());
    EXPECT_EQ(milvus::DataType::FLOAT_VECTOR, schema.Fields().at(2).FieldDataType());

    // has non-existent collection
    milvus::HasCollectionResponse has_resp2;
    status = client_->HasCollection(milvus::HasCollectionRequest().WithCollectionName("non_existent_collection_12345"),
                                    has_resp2);
    milvus::test::ExpectStatusOK(status);
    EXPECT_FALSE(has_resp2.Has());
}

TEST_F(MilvusServerTestCollectionOps, GetLoadState) {
    // before loading
    milvus::GetLoadStateResponse state_resp;
    auto status = client_->GetLoadState(milvus::GetLoadStateRequest().WithCollectionName(collection_name), state_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(milvus::LoadState::LOAD_STATE_NOT_LOAD, state_resp.State());

    // create index and load
    milvus::IndexDesc index_desc("vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    status = client_->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
    milvus::test::ExpectStatusOK(status);

    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    // after loading
    milvus::GetLoadStateResponse state_resp2;
    status = client_->GetLoadState(milvus::GetLoadStateRequest().WithCollectionName(collection_name), state_resp2);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(milvus::LoadState::LOAD_STATE_LOADED, state_resp2.State());

    // release
    status = client_->ReleaseCollection(milvus::ReleaseCollectionRequest().WithCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);
}

TEST_F(MilvusServerTestCollectionOps, FlushAndGetStats) {
    // insert some data
    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"Alice"}),
        std::make_shared<milvus::FloatVecFieldData>("vec", std::vector<std::vector<float>>{{0.1f, 0.2f, 0.3f, 0.4f}})};

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithColumnsData(std::move(fields));
    milvus::InsertResponse insert_resp;
    auto status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);

    // flush
    status = client_->Flush(milvus::FlushRequest().AddCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    // get stats
    milvus::GetCollectionStatsResponse stats_resp;
    status = client_->GetCollectionStats(milvus::GetCollectionStatsRequest().WithCollectionName(collection_name),
                                         stats_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(stats_resp.Stats().RowCount(), 1);
}

TEST_F(MilvusServerTestCollectionOps, AddCollectionField) {
    // add a new scalar field to the collection
    auto new_field = milvus::FieldSchema("age", milvus::DataType::INT16, "age field").WithNullable(true);
    auto status = client_->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithCollectionName(collection_name).WithField(std::move(new_field)));
    milvus::test::ExpectStatusOK(status);

    // verify the field was added by describing the collection
    milvus::DescribeCollectionResponse desc_resp;
    status =
        client_->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name), desc_resp);
    milvus::test::ExpectStatusOK(status);

    auto schema = desc_resp.Desc().Schema();
    EXPECT_EQ(4, schema.Fields().size());
    EXPECT_EQ(milvus::DataType::INT16, schema.Fields().at(3).FieldDataType());
}

TEST_F(MilvusServerTestCollectionOps, AlterAndDropCollectionProperties) {
    // alter collection properties
    auto status = client_->AlterCollectionProperties(milvus::AlterCollectionPropertiesRequest()
                                                         .WithCollectionName(collection_name)
                                                         .AddProperty("collection.ttl.seconds", "86400"));
    milvus::test::ExpectStatusOK(status);

    // verify by describing the collection
    milvus::DescribeCollectionResponse desc_resp;
    status =
        client_->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name), desc_resp);
    milvus::test::ExpectStatusOK(status);

    auto props = desc_resp.Desc().Properties();
    EXPECT_GE(props.size(), 1);
    EXPECT_TRUE(props.find("collection.ttl.seconds") != props.end());
    EXPECT_EQ(props["collection.ttl.seconds"], "86400");

    // drop collection properties
    status = client_->DropCollectionProperties(milvus::DropCollectionPropertiesRequest()
                                                   .WithCollectionName(collection_name)
                                                   .AddPropertyKey("collection.ttl.seconds"));
    milvus::test::ExpectStatusOK(status);

    // verify again by describing the collection
    status =
        client_->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name), desc_resp);
    milvus::test::ExpectStatusOK(status);

    props = desc_resp.Desc().Properties();
    EXPECT_TRUE(props.find("collection.ttl.seconds") == props.end());
}

TEST_F(MilvusServerTestCollectionOps, AlterAndDropCollectionFieldProperties) {
    // alter field properties (set max_length for varchar field)
    auto status = client_->AlterCollectionFieldProperties(milvus::AlterCollectionFieldPropertiesRequest()
                                                              .WithCollectionName(collection_name)
                                                              .WithFieldName("name")
                                                              .AddProperty(milvus::MMAP_ENABLED, "true"));
    milvus::test::ExpectStatusOK(status);

    // get field properties in DescribeCollectionResponse and verify the altered property is present and correct
    milvus::DescribeCollectionResponse desc_resp;
    status =
        client_->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name), desc_resp);
    milvus::test::ExpectStatusOK(status);

    auto schema = desc_resp.Desc().Schema();
    auto fields = schema.Fields();
    for (const auto& field : fields) {
        if (field.Name() == "name") {
            auto props = field.TypeParams();
            EXPECT_TRUE(props.find(milvus::MMAP_ENABLED) != props.end());
            EXPECT_EQ(props.at(milvus::MMAP_ENABLED), "true");
        }
    }

    // drop field properties
    status = client_->DropCollectionFieldProperties(milvus::DropCollectionFieldPropertiesRequest()
                                                        .WithCollectionName(collection_name)
                                                        .WithFieldName("name")
                                                        .AddPropertyKey(milvus::MMAP_ENABLED));
    milvus::test::ExpectStatusOK(status);

    // verify again by describing the collection
    milvus::DescribeCollectionResponse desc_resp2;
    status =
        client_->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name), desc_resp);
    milvus::test::ExpectStatusOK(status);

    schema = desc_resp.Desc().Schema();
    fields = schema.Fields();
    for (const auto& field : fields) {
        if (field.Name() == "name") {
            auto props = field.TypeParams();
            EXPECT_TRUE(props.find(milvus::MMAP_ENABLED) == props.end());
        }
    }
}

TEST_F(MilvusServerTestCollectionOps, TruncateCollection) {
    // create index first
    milvus::IndexDesc index_desc("vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    auto status = client_->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_desc)));
    milvus::test::ExpectStatusOK(status);

    // insert some data
    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"Alice", "Bob", "Charlie"}),
        std::make_shared<milvus::FloatVecFieldData>(
            "vec", std::vector<std::vector<float>>{
                       {0.1f, 0.2f, 0.3f, 0.4f}, {0.5f, 0.6f, 0.7f, 0.8f}, {0.9f, 1.0f, 1.1f, 1.2f}})};

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(collection_name).WithColumnsData(std::move(fields));
    milvus::InsertResponse insert_resp;
    status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 3);

    // load and verify data exists
    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    milvus::QueryRequest query_req;
    query_req.WithCollectionName(collection_name);
    query_req.WithFilter("id >= 0");
    query_req.AddOutputField("name");
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(query_resp.Results().GetRowCount(), 3);

    // release before truncate
    status = client_->ReleaseCollection(milvus::ReleaseCollectionRequest().WithCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    // truncate the collection
    status = client_->TruncateCollection(milvus::TruncateCollectionRequest().WithCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    // reload and verify data is gone
    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    milvus::test::ExpectStatusOK(status);

    milvus::QueryResponse query_resp2;
    status = client_->Query(query_req, query_resp2);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(query_resp2.Results().GetRowCount(), 0);
}

TEST_F(MilvusServerTestCollectionOps, DynamicField) {
    // create a separate collection with dynamic field enabled
    std::string dyn_coll = milvus::test::RanName("DynField_");

    auto schema = std::make_shared<milvus::CollectionSchema>(dyn_coll);
    schema->SetEnableDynamicField(true);
    schema->AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, false));
    schema->AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(4));
    schema->AddField(milvus::FieldSchema("text", milvus::DataType::VARCHAR, "text").WithMaxLength(256));

    auto status = client_->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(dyn_coll).WithCollectionSchema(schema));
    milvus::test::ExpectStatusOK(status);

    milvus::IndexDesc idx("vec", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    status = client_->CreateIndex(milvus::CreateIndexRequest().WithCollectionName(dyn_coll).AddIndex(std::move(idx)));
    milvus::test::ExpectStatusOK(status);

    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(dyn_coll));
    milvus::test::ExpectStatusOK(status);

    // insert with row-based — include dynamic fields "color" and "score"
    milvus::EntityRows rows_data;
    rows_data.push_back(nlohmann::json{
        {"id", 0},
        {"vec", {0.1f, 0.2f, 0.3f, 0.4f}},
        {"text", "hello"},
        {"color", "red"},
        {"score", 95},
    });
    rows_data.push_back(nlohmann::json{
        {"id", 1},
        {"vec", {0.5f, 0.6f, 0.7f, 0.8f}},
        {"text", "world"},
        {"color", "blue"},
        {"score", 80},
    });
    rows_data.push_back(nlohmann::json{
        {"id", 2},
        {"vec", {0.9f, 1.0f, 1.1f, 1.2f}},
        {"text", "foo"},
        {"color", "red"},
        {"score", 60},
    });

    milvus::InsertRequest insert_req;
    insert_req.WithCollectionName(dyn_coll).WithRowsData(std::move(rows_data));
    milvus::InsertResponse insert_resp;
    status = client_->Insert(insert_req, insert_resp);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_resp.Results().InsertCount(), 3);

    // query and output dynamic fields
    milvus::QueryRequest query_req;
    query_req.WithCollectionName(dyn_coll);
    query_req.WithFilter("id >= 0");
    query_req.AddOutputField("text");
    query_req.AddOutputField("color");
    query_req.AddOutputField("score");
    query_req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse query_resp;
    status = client_->Query(query_req, query_resp);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows rows;
    query_resp.Results().OutputRows(rows);
    EXPECT_EQ(rows.size(), 3);

    std::map<int64_t, nlohmann::json> result_map;
    for (const auto& row : rows) {
        result_map[row["id"].get<int64_t>()] = row;
    }

    EXPECT_EQ(result_map[0]["color"].get<std::string>(), "red");
    EXPECT_EQ(result_map[0]["score"].get<int>(), 95);
    EXPECT_EQ(result_map[1]["color"].get<std::string>(), "blue");
    EXPECT_EQ(result_map[1]["score"].get<int>(), 80);
    EXPECT_EQ(result_map[2]["color"].get<std::string>(), "red");
    EXPECT_EQ(result_map[2]["score"].get<int>(), 60);

    // search with filter on dynamic field
    milvus::SearchRequest search_req;
    search_req.WithCollectionName(dyn_coll);
    search_req.WithAnnsField("vec");
    search_req.WithLimit(10);
    search_req.AddFloatVector({0.5f, 0.5f, 0.5f, 0.5f});
    search_req.WithFilter(R"(color == "red")");
    search_req.AddOutputField("color");
    search_req.AddOutputField("score");
    search_req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResponse search_resp;
    status = client_->Search(search_req, search_resp);
    milvus::test::ExpectStatusOK(status);

    const auto& results = search_resp.Results().Results();
    EXPECT_EQ(results.size(), 1);
    EXPECT_EQ(results.at(0).Scores().size(), 2);  // only rows 0 and 2

    milvus::EntityRows search_rows;
    results.at(0).OutputRows(search_rows);
    for (const auto& row : search_rows) {
        EXPECT_EQ(row["color"].get<std::string>(), "red");
    }

    // cleanup
    client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(dyn_coll));
}
