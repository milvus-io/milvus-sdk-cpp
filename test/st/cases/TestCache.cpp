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
#include <cstdlib>
#include <thread>
#include <vector>

#include "MilvusServerTest.h"
#include "utils/cache/CollectionTsCache.h"
#include "utils/cache/SchemaCache.h"

class MilvusServerTestCache : public milvus::test::MilvusServerTest {
 protected:
    std::string collection_name_;
    std::string endpoint_;
    std::vector<std::string> extra_collections_;
    std::vector<std::string> aliases_;

    void
    CreateAndLoadCollection(const std::string& collection_name, const std::string& primary_field = "id") {
        auto schema = std::make_shared<milvus::CollectionSchema>(collection_name);
        schema->SetEnableDynamicField(false);
        schema->AddField(milvus::FieldSchema(primary_field, milvus::DataType::INT64, primary_field, true, false));
        schema->AddField(milvus::FieldSchema("text", milvus::DataType::VARCHAR, "text").WithMaxLength(128));
        schema->AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR, "vector").WithDimension(2));

        auto status = client_->CreateCollection(
            milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
        milvus::test::ExpectStatusOK(status);

        status = client_->CreateIndex(
            milvus::CreateIndexRequest()
                .WithCollectionName(collection_name)
                .AddIndex(milvus::IndexDesc("vector", "", milvus::IndexType::FLAT, milvus::MetricType::L2)));
        milvus::test::ExpectStatusOK(status);

        status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
        milvus::test::ExpectStatusOK(status);
    }

    void
    SetUp() override {
        MilvusServerTest::SetUp();
        milvus::SchemaCache::GetInstance().Clear();
        milvus::CollectionTsCache::GetInstance().Clear();

        const char* host = std::getenv("MILVUS_HOST");
        endpoint_ = std::string(host ? host : "localhost") + ":29730";
        collection_name_ = milvus::test::RanName("CacheTest_");
        CreateAndLoadCollection(collection_name_);
    }

    void
    TearDown() override {
        for (const auto& alias : aliases_) {
            client_->DropAlias(milvus::DropAliasRequest().WithAlias(alias));
        }
        for (const auto& collection : extra_collections_) {
            client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection));
        }
        client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name_));
        milvus::SchemaCache::GetInstance().Clear();
        milvus::CollectionTsCache::GetInstance().Clear();
        MilvusServerTest::TearDown();
    }

    static milvus::EntityRow
    MakeRow(int64_t id) {
        return nlohmann::json{{"id", id},
                              {"text", "text_" + std::to_string(id)},
                              {"vector", {static_cast<float>(id), static_cast<float>(id + 1)}}};
    }
};

TEST_F(MilvusServerTestCache, ConcurrentDmlPopulatesSharedCaches) {
    constexpr int kThreadCount = 8;
    milvus::SchemaCache::GetInstance().Clear();
    milvus::CollectionTsCache::GetInstance().Clear();

    std::vector<milvus::Status> statuses(kThreadCount);
    std::vector<milvus::InsertResponse> responses(kThreadCount);
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([&, i]() {
            milvus::EntityRows rows{MakeRow(i)};
            statuses[i] = client_->Insert(
                milvus::InsertRequest().WithCollectionName(collection_name_).WithRowsData(std::move(rows)),
                responses[i]);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    uint64_t latest_timestamp = 0;
    for (int i = 0; i < kThreadCount; ++i) {
        milvus::test::ExpectStatusOK(statuses[i]);
        EXPECT_EQ(responses[i].Results().InsertCount(), 1);
        latest_timestamp = std::max(latest_timestamp, responses[i].Results().Timestamp());
    }

    milvus::CollectionDescPtr desc;
    ASSERT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", collection_name_, desc));
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->CollectionName(), collection_name_);
    EXPECT_EQ(desc->Schema().Fields().size(), 3);
    EXPECT_EQ(milvus::SchemaCache::GetInstance().Size(), 1);

    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", collection_name_), latest_timestamp);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Size(), 1);
}

TEST_F(MilvusServerTestCache, SessionQueryUsesLatestCollectionTimestamp) {
    milvus::EntityRows rows{MakeRow(1), MakeRow(2), MakeRow(3)};
    milvus::InsertResponse insert_response;
    auto status = client_->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name_).WithRowsData(std::move(rows)), insert_response);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(insert_response.Results().InsertCount(), 3);

    const auto cached_timestamp = milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", collection_name_);
    EXPECT_EQ(cached_timestamp, insert_response.Results().Timestamp());
    EXPECT_GT(cached_timestamp, 0);

    milvus::QueryResponse query_response;
    status = client_->Query(milvus::QueryRequest()
                                .WithCollectionName(collection_name_)
                                .WithFilter("id >= 0")
                                .AddOutputField("id")
                                .AddOutputField("text")
                                .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION),
                            query_response);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows output_rows;
    status = query_response.Results().OutputRows(output_rows);
    milvus::test::ExpectStatusOK(status);
    ASSERT_EQ(output_rows.size(), 3);

    std::vector<int64_t> ids;
    ids.reserve(output_rows.size());
    for (const auto& row : output_rows) {
        ids.emplace_back(row.at("id").get<int64_t>());
    }
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(ids, (std::vector<int64_t>{1, 2, 3}));
}

TEST_F(MilvusServerTestCache, RefreshesStaleSchemaAfterAddingField) {
    milvus::InsertResponse insert_response;
    milvus::EntityRows rows{MakeRow(1)};
    auto status = client_->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name_).WithRowsData(std::move(rows)), insert_response);
    milvus::test::ExpectStatusOK(status);

    milvus::CollectionDescPtr old_desc;
    ASSERT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", collection_name_, old_desc));
    ASSERT_NE(old_desc, nullptr);
    EXPECT_EQ(old_desc->Schema().Fields().size(), 3);

    auto extra_field = milvus::FieldSchema("extra", milvus::DataType::INT64, "extra").WithNullable(true);
    status = client_->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithCollectionName(collection_name_).WithField(std::move(extra_field)));
    milvus::test::ExpectStatusOK(status);

    rows = {nlohmann::json{{"id", 2}, {"text", "text_2"}, {"vector", {2.0f, 3.0f}}, {"extra", 20}}};
    status = client_->Insert(milvus::InsertRequest().WithCollectionName(collection_name_).WithRowsData(std::move(rows)),
                             insert_response);
    milvus::test::ExpectStatusOK(status);

    milvus::CollectionDescPtr refreshed_desc;
    ASSERT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", collection_name_, refreshed_desc));
    ASSERT_NE(refreshed_desc, nullptr);
    const auto& fields = refreshed_desc->Schema().Fields();
    const auto extra = std::find_if(fields.begin(), fields.end(),
                                    [](const milvus::FieldSchema& field) { return field.Name() == "extra"; });
    ASSERT_NE(extra, fields.end());
    EXPECT_EQ(extra->FieldDataType(), milvus::DataType::INT64);

    milvus::QueryResponse query_response;
    status = client_->Query(milvus::QueryRequest()
                                .WithCollectionName(collection_name_)
                                .WithFilter("id == 2")
                                .AddOutputField("extra")
                                .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION),
                            query_response);
    milvus::test::ExpectStatusOK(status);

    milvus::EntityRows output_rows;
    status = query_response.Results().OutputRows(output_rows);
    milvus::test::ExpectStatusOK(status);
    ASSERT_EQ(output_rows.size(), 1);
    EXPECT_EQ(output_rows.front().at("extra").get<int64_t>(), 20);
}

TEST_F(MilvusServerTestCache, UpsertDeleteAndGetMaintainCaches) {
    milvus::InsertResponse insert_response;
    milvus::EntityRows rows{MakeRow(1)};
    auto status = client_->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name_).WithRowsData(std::move(rows)), insert_response);
    milvus::test::ExpectStatusOK(status);

    milvus::UpsertResponse upsert_response;
    rows = {nlohmann::json{{"id", 1}, {"text", "updated"}, {"vector", {10.0f, 11.0f}}}};
    status = client_->Upsert(milvus::UpsertRequest().WithCollectionName(collection_name_).WithRowsData(std::move(rows)),
                             upsert_response);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(upsert_response.Results().UpsertCount(), 1);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", collection_name_),
              upsert_response.Results().Timestamp());

    milvus::SchemaCache::GetInstance().Clear();
    milvus::GetResponse get_response;
    status = client_->Get(milvus::GetRequest()
                              .WithCollectionName(collection_name_)
                              .WithIDs(std::vector<int64_t>{1})
                              .AddOutputField("text")
                              .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION),
                          get_response);
    milvus::test::ExpectStatusOK(status);
    ASSERT_EQ(get_response.Results().GetRowCount(), 1);

    milvus::EntityRows output_rows;
    status = get_response.Results().OutputRows(output_rows);
    milvus::test::ExpectStatusOK(status);
    ASSERT_EQ(output_rows.size(), 1);
    EXPECT_EQ(output_rows.front().at("text").get<std::string>(), "updated");

    milvus::CollectionDescPtr desc;
    ASSERT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", collection_name_, desc));
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->Schema().PrimaryFieldName(), "id");

    milvus::DeleteResponse delete_response;
    status = client_->Delete(
        milvus::DeleteRequest().WithCollectionName(collection_name_).WithIDs(std::vector<int64_t>{1}), delete_response);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(delete_response.Results().DeleteCount(), 1);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", collection_name_),
              delete_response.Results().Timestamp());

    milvus::QueryResponse query_response;
    status = client_->Query(milvus::QueryRequest()
                                .WithCollectionName(collection_name_)
                                .WithFilter("id == 1")
                                .AddOutputField("id")
                                .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION),
                            query_response);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(query_response.Results().GetRowCount(), 0);
}

TEST_F(MilvusServerTestCache, IteratorsUseSessionTimestampWithExpectedSchemaCacheBehavior) {
    milvus::InsertResponse insert_response;
    milvus::EntityRows rows{MakeRow(10), MakeRow(11), MakeRow(12)};
    auto status = client_->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name_).WithRowsData(std::move(rows)), insert_response);
    milvus::test::ExpectStatusOK(status);
    const auto timestamp = insert_response.Results().Timestamp();
    ASSERT_GT(timestamp, 0);

    milvus::SchemaCache::GetInstance().Clear();
    milvus::QueryIteratorRequest query_request;
    query_request.WithCollectionName(collection_name_)
        .WithFilter("id >= 10")
        .AddOutputField("id")
        .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION);
    query_request.SetBatchSize(2);
    query_request.SetLimit(1);

    milvus::QueryIteratorPtr query_iterator;
    status = client_->QueryIterator(query_request, query_iterator);
    milvus::test::ExpectStatusOK(status);
    ASSERT_NE(query_iterator, nullptr);

    milvus::CollectionDescPtr desc;
    ASSERT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", collection_name_, desc));
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", collection_name_), timestamp);

    milvus::QueryResults query_batch;
    status = query_iterator->Next(query_batch);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(query_batch.GetRowCount(), 1);

    milvus::SchemaCache::GetInstance().Clear();
    milvus::SearchIteratorRequest search_request;
    search_request.WithCollectionName(collection_name_)
        .WithAnnsField("vector")
        .AddFloatVector({10.0f, 11.0f})
        .AddOutputField("id")
        .WithMetricType(milvus::MetricType::L2)
        .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION)
        .WithLimit(1);
    search_request.SetBatchSize(2);

    milvus::SearchIteratorPtr search_iterator;
    status = client_->SearchIterator(search_request, search_iterator);
    milvus::test::ExpectStatusOK(status);
    ASSERT_NE(search_iterator, nullptr);
    milvus::CollectionDescPtr search_desc;
    EXPECT_FALSE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", collection_name_, search_desc));
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", collection_name_), timestamp);

    milvus::SingleResult search_batch;
    status = search_iterator->Next(search_batch);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(search_batch.GetRowCount(), 1);
}

TEST_F(MilvusServerTestCache, AliasMutationsUpdateSchemaAndTimestampCaches) {
    const auto alias = milvus::test::RanName("CacheAlias_");
    const auto second_collection = milvus::test::RanName("CacheAliasTarget_");
    aliases_.push_back(alias);
    extra_collections_.push_back(second_collection);
    CreateAndLoadCollection(second_collection, "pk");

    milvus::InsertResponse first_insert;
    milvus::EntityRows rows{MakeRow(1)};
    auto status = client_->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name_).WithRowsData(std::move(rows)), first_insert);
    milvus::test::ExpectStatusOK(status);

    milvus::InsertResponse second_insert;
    rows = {nlohmann::json{{"pk", 100}, {"text", "second"}, {"vector", {100.0f, 101.0f}}}};
    status = client_->Insert(
        milvus::InsertRequest().WithCollectionName(second_collection).WithRowsData(std::move(rows)), second_insert);
    milvus::test::ExpectStatusOK(status);

    status = client_->CreateAlias(milvus::CreateAliasRequest().WithCollectionName(collection_name_).WithAlias(alias));
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", alias), first_insert.Results().Timestamp());

    milvus::GetResponse get_response;
    status = client_->Get(milvus::GetRequest()
                              .WithCollectionName(alias)
                              .WithIDs(std::vector<int64_t>{1})
                              .AddOutputField("text")
                              .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION),
                          get_response);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(get_response.Results().GetRowCount(), 1);

    milvus::CollectionDescPtr alias_desc;
    ASSERT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", alias, alias_desc));
    ASSERT_NE(alias_desc, nullptr);
    EXPECT_EQ(alias_desc->Schema().PrimaryFieldName(), "id");

    status = client_->AlterAlias(milvus::AlterAliasRequest().WithCollectionName(second_collection).WithAlias(alias));
    milvus::test::ExpectStatusOK(status);
    EXPECT_FALSE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", alias, alias_desc));
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", alias),
              std::max(first_insert.Results().Timestamp(), second_insert.Results().Timestamp()));

    status = client_->Get(milvus::GetRequest()
                              .WithCollectionName(alias)
                              .WithIDs(std::vector<int64_t>{100})
                              .AddOutputField("text")
                              .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION),
                          get_response);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(get_response.Results().GetRowCount(), 1);
    ASSERT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", alias, alias_desc));
    ASSERT_NE(alias_desc, nullptr);
    EXPECT_EQ(alias_desc->Schema().PrimaryFieldName(), "pk");

    status = client_->DropAlias(milvus::DropAliasRequest().WithAlias(alias));
    milvus::test::ExpectStatusOK(status);
    EXPECT_FALSE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", alias, alias_desc));
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", alias), 0);
}

TEST_F(MilvusServerTestCache, RenameAndTruncateUpdateCaches) {
    milvus::InsertResponse insert_response;
    milvus::EntityRows rows{MakeRow(1)};
    auto status = client_->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name_).WithRowsData(std::move(rows)), insert_response);
    milvus::test::ExpectStatusOK(status);

    const auto old_name = collection_name_;
    const auto new_name = milvus::test::RanName("CacheRenamed_");
    const auto timestamp = insert_response.Results().Timestamp();
    status = client_->RenameCollection(
        milvus::RenameCollectionRequest().WithCollectionName(old_name).WithNewCollectionName(new_name));
    milvus::test::ExpectStatusOK(status);
    ASSERT_TRUE(status.IsOk());
    collection_name_ = new_name;

    milvus::CollectionDescPtr desc;
    EXPECT_FALSE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", old_name, desc));
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", old_name), 0);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", new_name), timestamp);

    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(new_name));
    milvus::test::ExpectStatusOK(status);

    milvus::GetResponse get_response;
    status = client_->Get(milvus::GetRequest()
                              .WithCollectionName(new_name)
                              .WithIDs(std::vector<int64_t>{1})
                              .AddOutputField("text")
                              .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION),
                          get_response);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(get_response.Results().GetRowCount(), 1);
    ASSERT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", new_name, desc));

    status = client_->ReleaseCollection(milvus::ReleaseCollectionRequest().WithCollectionName(new_name));
    milvus::test::ExpectStatusOK(status);
    status = client_->TruncateCollection(milvus::TruncateCollectionRequest().WithCollectionName(new_name));
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", new_name), 0);
    EXPECT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", new_name, desc));

    status = client_->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(new_name));
    milvus::test::ExpectStatusOK(status);
    milvus::QueryResponse query_response;
    status = client_->Query(milvus::QueryRequest()
                                .WithCollectionName(new_name)
                                .WithFilter("id >= 0")
                                .AddOutputField("id")
                                .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION),
                            query_response);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(query_response.Results().GetRowCount(), 0);
}

TEST_F(MilvusServerTestCache, DropAndRecreateCollectionDoesNotReuseOldSchema) {
    milvus::InsertResponse insert_response;
    milvus::EntityRows rows{MakeRow(1)};
    auto status = client_->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name_).WithRowsData(std::move(rows)), insert_response);
    milvus::test::ExpectStatusOK(status);

    milvus::CollectionDescPtr desc;
    ASSERT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", collection_name_, desc));
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->Schema().PrimaryFieldName(), "id");
    EXPECT_GT(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", collection_name_), 0);

    status = client_->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name_));
    milvus::test::ExpectStatusOK(status);
    EXPECT_FALSE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", collection_name_, desc));
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint_, "", collection_name_), 0);

    CreateAndLoadCollection(collection_name_, "pk");
    rows = {nlohmann::json{{"pk", 100}, {"text", "recreated"}, {"vector", {100.0f, 101.0f}}}};
    status = client_->Insert(milvus::InsertRequest().WithCollectionName(collection_name_).WithRowsData(std::move(rows)),
                             insert_response);
    milvus::test::ExpectStatusOK(status);

    ASSERT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint_, "", collection_name_, desc));
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->Schema().PrimaryFieldName(), "pk");

    milvus::GetResponse get_response;
    status = client_->Get(milvus::GetRequest()
                              .WithCollectionName(collection_name_)
                              .WithIDs(std::vector<int64_t>{100})
                              .AddOutputField("text")
                              .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION),
                          get_response);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(get_response.Results().GetRowCount(), 1);
}
