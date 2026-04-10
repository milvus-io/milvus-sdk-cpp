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
#include "milvus/request/dql/QueryIteratorRequest.h"

class QueryRequestTest : public ::testing::Test {};

TEST_F(QueryRequestTest, GettersAndSetters) {
    milvus::QueryRequest req;

    req.WithCollectionName("query_coll");
    EXPECT_EQ(req.CollectionName(), "query_coll");

    req.AddPartitionName("p1");
    EXPECT_EQ(req.PartitionNames().size(), 1);
    EXPECT_TRUE(req.PartitionNames().count("p1"));

    req.AddOutputField("field1");
    req.AddOutputField("field2");
    EXPECT_EQ(req.OutputFields().size(), 2);

    req.WithFilter("id > 10");
    EXPECT_EQ(req.Filter(), "id > 10");

    req.WithLimit(100);
    EXPECT_EQ(req.Limit(), 100);

    req.WithOffset(50);
    EXPECT_EQ(req.Offset(), 50);

    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    EXPECT_EQ(req.GetConsistencyLevel(), milvus::ConsistencyLevel::STRONG);

    nlohmann::json tmpl = 10;
    req.AddFilterTemplate("threshold", tmpl);
    EXPECT_EQ(req.FilterTemplates().size(), 1);

    req.WithIgnoreGrowing(true);
    EXPECT_TRUE(req.IgnoreGrowing());

    req.AddExtraParam("hints", "segcore");
    EXPECT_EQ(req.ExtraParams().at("hints"), "segcore");
}

TEST_F(QueryRequestTest, DQLRequestBaseMethods) {
    milvus::QueryRequest req;

    // DatabaseName
    EXPECT_TRUE(req.DatabaseName().empty());
    req.SetDatabaseName("test_db");
    EXPECT_EQ(req.DatabaseName(), "test_db");

    milvus::QueryRequest req2;
    req2.WithDatabaseName("db2");
    EXPECT_EQ(req2.DatabaseName(), "db2");

    // WithPartitionNames (bulk set)
    std::set<std::string> parts{"p1", "p2", "p3"};
    req.WithPartitionNames(std::move(parts));
    EXPECT_EQ(req.PartitionNames().size(), 3);
    EXPECT_TRUE(req.PartitionNames().count("p2"));

    // SetPartitionNames
    std::set<std::string> parts2{"x1"};
    req.SetPartitionNames(std::move(parts2));
    EXPECT_EQ(req.PartitionNames().size(), 1);
    EXPECT_TRUE(req.PartitionNames().count("x1"));

    // SetOutputFields / WithOutputFields (bulk set)
    std::set<std::string> fields{"f1", "f2"};
    req.SetOutputFields(std::move(fields));
    EXPECT_EQ(req.OutputFields().size(), 2);
    EXPECT_TRUE(req.OutputFields().count("f1"));

    std::set<std::string> fields2{"a", "b", "c"};
    req.WithOutputFields(std::move(fields2));
    EXPECT_EQ(req.OutputFields().size(), 3);
    EXPECT_TRUE(req.OutputFields().count("c"));

    // SetConsistencyLevel
    req.SetConsistencyLevel(milvus::ConsistencyLevel::SESSION);
    EXPECT_EQ(req.GetConsistencyLevel(), milvus::ConsistencyLevel::SESSION);

    // SetCollectionName
    req.SetCollectionName("another_coll");
    EXPECT_EQ(req.CollectionName(), "another_coll");
}

class GetRequestTest : public ::testing::Test {};

TEST_F(GetRequestTest, GettersAndSetters) {
    milvus::GetRequest req;

    req.WithCollectionName("get_coll");
    EXPECT_EQ(req.CollectionName(), "get_coll");

    req.AddOutputField("name");
    EXPECT_EQ(req.OutputFields().size(), 1);

    // Int64 IDs
    std::vector<int64_t> int_ids{1, 2, 3};
    req.WithIDs(std::move(int_ids));
    EXPECT_TRUE(req.IDs().IsIntegerID());
    EXPECT_EQ(req.IDs().IntIDArray().size(), 3);

    // String IDs
    milvus::GetRequest req2;
    std::vector<std::string> str_ids{"a", "b"};
    req2.WithIDs(std::move(str_ids));
    EXPECT_FALSE(req2.IDs().IsIntegerID());
    EXPECT_EQ(req2.IDs().StrIDArray().size(), 2);

    req.WithConsistencyLevel(milvus::ConsistencyLevel::SESSION);
    EXPECT_EQ(req.GetConsistencyLevel(), milvus::ConsistencyLevel::SESSION);
}

TEST_F(GetRequestTest, DQLRequestBaseMethods) {
    milvus::GetRequest req;

    req.WithDatabaseName("db1");
    EXPECT_EQ(req.DatabaseName(), "db1");

    req.SetDatabaseName("db2");
    EXPECT_EQ(req.DatabaseName(), "db2");

    std::set<std::string> parts{"p1", "p2"};
    req.WithPartitionNames(std::move(parts));
    EXPECT_EQ(req.PartitionNames().size(), 2);

    std::set<std::string> fields{"f1", "f2"};
    req.SetOutputFields(std::move(fields));
    EXPECT_EQ(req.OutputFields().size(), 2);

    std::set<std::string> fields2{"a"};
    req.WithOutputFields(std::move(fields2));
    EXPECT_EQ(req.OutputFields().size(), 1);
}

class SearchRequestTest : public ::testing::Test {};

TEST_F(SearchRequestTest, GettersAndSetters) {
    milvus::SearchRequest req;

    req.WithCollectionName("search_coll");
    EXPECT_EQ(req.CollectionName(), "search_coll");

    req.AddPartitionName("p1");
    EXPECT_EQ(req.PartitionNames().size(), 1);

    req.AddOutputField("field1");
    EXPECT_EQ(req.OutputFields().size(), 1);

    req.WithFilter("color == 'red'");
    EXPECT_EQ(req.Filter(), "color == 'red'");

    req.WithLimit(10);
    EXPECT_EQ(req.Limit(), 10);

    req.WithAnnsField("embedding");
    EXPECT_EQ(req.AnnsField(), "embedding");

    req.WithMetricType(milvus::MetricType::L2);
    EXPECT_EQ(req.MetricType(), milvus::MetricType::L2);

    req.AddExtraParam("nprobe", "16");
    EXPECT_EQ(req.ExtraParams().at("nprobe"), "16");

    req.WithRangeFilter(0.5);
    EXPECT_NEAR(req.RangeFilter(), 0.5, 0.001);

    req.WithRadius(1.0);
    EXPECT_NEAR(req.Radius(), 1.0, 0.001);

    req.WithOffset(5);
    EXPECT_EQ(req.Offset(), 5);

    req.WithGroupByField("category");
    EXPECT_EQ(req.GroupByField(), "category");

    req.WithGroupSize(3);
    EXPECT_EQ(req.GroupSize(), 3);

    req.WithStrictGroupSize(true);
    EXPECT_TRUE(req.StrictGroupSize());

    req.WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);
    EXPECT_EQ(req.GetConsistencyLevel(), milvus::ConsistencyLevel::BOUNDED);

    nlohmann::json tmpl = "red";
    req.AddFilterTemplate("color_val", tmpl);
    EXPECT_EQ(req.FilterTemplates().size(), 1);

    req.WithIgnoreGrowing(true);
    EXPECT_TRUE(req.IgnoreGrowing());

    // AddFloatVector
    std::vector<float> float_vec{1.0f, 2.0f, 3.0f};
    req.AddFloatVector(float_vec);
    EXPECT_NE(req.TargetVectors(), nullptr);

    // AddBinaryVector
    milvus::SearchRequest req2;
    std::vector<uint8_t> bin_vec{0x01, 0x02};
    req2.AddBinaryVector(bin_vec);
    EXPECT_NE(req2.TargetVectors(), nullptr);
}

TEST_F(SearchRequestTest, WithExtraParams) {
    milvus::SearchRequest req;
    std::unordered_map<std::string, std::string> params{{"nprobe", "16"}, {"ef", "64"}};
    auto& ref = req.WithExtraParams(params);
    EXPECT_EQ(&ref, &req);
    EXPECT_EQ(req.ExtraParams().at("nprobe"), "16");
    EXPECT_EQ(req.ExtraParams().at("ef"), "64");
}

TEST_F(SearchRequestTest, WithRoundDecimal) {
    milvus::SearchRequest req;
    auto& ref = req.WithRoundDecimal(3);
    EXPECT_EQ(req.RoundDecimal(), 3);
    EXPECT_EQ(&ref, &req);
}

TEST_F(SearchRequestTest, SetGroupByField) {
    milvus::SearchRequest req;
    req.SetGroupByField("category");
    EXPECT_EQ(req.GroupByField(), "category");
}

TEST_F(SearchRequestTest, DQLRequestBaseMethods) {
    milvus::SearchRequest req;

    req.WithDatabaseName("db1");
    EXPECT_EQ(req.DatabaseName(), "db1");

    req.SetDatabaseName("db2");
    EXPECT_EQ(req.DatabaseName(), "db2");

    std::set<std::string> parts{"p1", "p2"};
    req.WithPartitionNames(std::move(parts));
    EXPECT_EQ(req.PartitionNames().size(), 2);

    std::set<std::string> fields{"f1", "f2"};
    req.SetOutputFields(std::move(fields));
    EXPECT_EQ(req.OutputFields().size(), 2);

    std::set<std::string> fields2{"a"};
    req.WithOutputFields(std::move(fields2));
    EXPECT_EQ(req.OutputFields().size(), 1);

    req.SetCollectionName("coll2");
    EXPECT_EQ(req.CollectionName(), "coll2");

    req.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);
    EXPECT_EQ(req.GetConsistencyLevel(), milvus::ConsistencyLevel::BOUNDED);
}

TEST_F(SearchRequestTest, SearchRequestBaseAllVectorTypes) {
    // Test all vector Add methods through SearchRequest (covers SearchRequestBase + SearchRequestVectorAssigner)

    // AddBinaryVector (string)
    {
        milvus::SearchRequest req;
        req.WithAnnsField("bin_field");
        std::string bin_str(4, '\xFF');
        req.AddBinaryVector(bin_str);
        EXPECT_NE(req.TargetVectors(), nullptr);
    }

    // AddBinaryVector (uint8)
    {
        milvus::SearchRequest req;
        req.WithAnnsField("bin_field");
        req.AddBinaryVector(std::vector<uint8_t>{0xFF, 0x00, 0xAB, 0xCD});
        EXPECT_NE(req.TargetVectors(), nullptr);
    }

    // AddFloat16Vector (binary)
    {
        milvus::SearchRequest req;
        req.WithAnnsField("fp16_field");
        req.AddFloat16Vector(std::vector<uint16_t>{0x3C00, 0x4000, 0x4200, 0x4400});
        EXPECT_NE(req.TargetVectors(), nullptr);
    }

    // AddFloat16Vector (float auto-convert)
    {
        milvus::SearchRequest req;
        req.WithAnnsField("fp16_field");
        req.AddFloat16Vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
        EXPECT_NE(req.TargetVectors(), nullptr);
    }

    // AddBFloat16Vector (binary)
    {
        milvus::SearchRequest req;
        req.WithAnnsField("bf16_field");
        req.AddBFloat16Vector(std::vector<uint16_t>{0x3F80, 0x4000});
        EXPECT_NE(req.TargetVectors(), nullptr);
    }

    // AddBFloat16Vector (float auto-convert)
    {
        milvus::SearchRequest req;
        req.WithAnnsField("bf16_field");
        req.AddBFloat16Vector(std::vector<float>{1.0f, 2.0f});
        EXPECT_NE(req.TargetVectors(), nullptr);
    }

    // AddSparseVector (map)
    {
        milvus::SearchRequest req;
        req.WithAnnsField("sparse_field");
        std::map<uint32_t, float> sparse = {{1, 0.1f}, {5, 0.2f}};
        req.AddSparseVector(sparse);
        EXPECT_NE(req.TargetVectors(), nullptr);
    }

    // AddSparseVector (json)
    {
        milvus::SearchRequest req;
        req.WithAnnsField("sparse_field");
        req.AddSparseVector(nlohmann::json{{"1", 0.1}, {"5", 0.2}});
        EXPECT_NE(req.TargetVectors(), nullptr);
    }

    // AddEmbeddedText
    {
        milvus::SearchRequest req;
        req.WithAnnsField("text_field");
        req.AddEmbeddedText("hello world");
        EXPECT_NE(req.TargetVectors(), nullptr);
    }

    // AddInt8Vector
    {
        milvus::SearchRequest req;
        req.WithAnnsField("int8_field");
        req.AddInt8Vector(std::vector<int8_t>{1, -2, 3, -4});
        EXPECT_NE(req.TargetVectors(), nullptr);
    }

    // AddEmbeddingList
    {
        milvus::SearchRequest req;
        req.WithAnnsField("struct_field");
        milvus::EmbeddingList emb;
        emb.AddFloatVector({1.0f, 2.0f, 3.0f});
        req.AddEmbeddingList(std::move(emb));
        EXPECT_EQ(req.EmbeddingLists().size(), 1);
    }
}

TEST_F(SearchRequestTest, SearchRequestBaseSetMethods) {
    milvus::SearchRequest req;

    // SetAnnsField
    req.SetAnnsField("vec_field");
    EXPECT_EQ(req.AnnsField(), "vec_field");

    // SetLimit
    req.SetLimit(50);
    EXPECT_EQ(req.Limit(), 50);

    // SetMetricType
    req.SetMetricType(milvus::MetricType::COSINE);
    EXPECT_EQ(req.MetricType(), milvus::MetricType::COSINE);

    // SetRadius / Radius
    req.SetRadius(1.5);
    EXPECT_DOUBLE_EQ(req.Radius(), 1.5);

    // SetRangeFilter / RangeFilter
    req.SetRangeFilter(0.5);
    EXPECT_DOUBLE_EQ(req.RangeFilter(), 0.5);

    // SetRange (sets both)
    req.SetRange(0.3, 1.0);
    EXPECT_DOUBLE_EQ(req.RangeFilter(), 0.3);
    EXPECT_DOUBLE_EQ(req.Radius(), 1.0);

    // SetTimezone / Timezone
    req.SetTimezone("UTC+8");
    EXPECT_EQ(req.Timezone(), "UTC+8");

    // Filter / SetFilter (via SearchRequestBase through SearchRequest)
    req.WithFilter("age > 10");
    EXPECT_EQ(req.Filter(), "age > 10");

    // SetFilterTemplates
    std::unordered_map<std::string, nlohmann::json> tmpls;
    tmpls["threshold"] = 100;
    req.WithFilterTemplates(std::move(tmpls));
    EXPECT_EQ(req.FilterTemplates().size(), 1);

    // Validate
    req.AddFloatVector({1.0f, 2.0f, 3.0f});
    auto status = req.Validate();
    EXPECT_TRUE(status.IsOk());
}

class HybridSearchRequestTest : public ::testing::Test {};

TEST_F(HybridSearchRequestTest, GettersAndSetters) {
    milvus::HybridSearchRequest req;

    req.WithCollectionName("hybrid_coll");
    EXPECT_EQ(req.CollectionName(), "hybrid_coll");

    // AddSubRequest
    auto sub = std::make_shared<milvus::SubSearchRequest>();
    sub->WithAnnsField("vec1").WithLimit(10);
    req.AddSubRequest(sub);
    EXPECT_EQ(req.SubRequests().size(), 1);

    // Rerank
    auto rerank = std::make_shared<milvus::RRFRerank>(60);
    req.WithRerank(rerank);
    EXPECT_NE(req.Rerank(), nullptr);

    // Limit
    req.WithLimit(20);
    EXPECT_EQ(req.Limit(), 20);

    // Offset
    req.WithOffset(10);
    EXPECT_EQ(req.Offset(), 10);

    // GroupByField
    req.WithGroupByField("category");
    EXPECT_EQ(req.GetGroupByField(), "category");

    // GroupSize
    req.WithGroupSize(5);
    EXPECT_EQ(req.GroupSize(), 5);

    // AddOutputField (inherited from DQLRequestBase)
    req.AddOutputField("field1");
    EXPECT_EQ(req.OutputFields().size(), 1);

    // AddExtraParam
    req.AddExtraParam("hints", "segcore");
    EXPECT_EQ(req.ExtraParams().at("hints"), "segcore");

    // ConsistencyLevel (inherited from DQLRequestBase)
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    EXPECT_EQ(req.GetConsistencyLevel(), milvus::ConsistencyLevel::STRONG);
}

TEST_F(HybridSearchRequestTest, WithRoundDecimal) {
    milvus::HybridSearchRequest req;
    auto& ref = req.WithRoundDecimal(4);
    EXPECT_EQ(req.GetRoundDecimal(), 4);
    EXPECT_EQ(&ref, &req);
}

TEST_F(HybridSearchRequestTest, WithStrictGroupSize) {
    milvus::HybridSearchRequest req;
    EXPECT_FALSE(req.StrictGroupSize());

    auto& ref = req.WithStrictGroupSize(true);
    EXPECT_TRUE(req.StrictGroupSize());
    EXPECT_EQ(&ref, &req);
}

class QueryIteratorRequestTest : public ::testing::Test {};

TEST_F(QueryIteratorRequestTest, SetReduceStopForBest) {
    milvus::QueryIteratorRequest req;
    EXPECT_FALSE(req.ReduceStopForBest());

    req.SetReduceStopForBest(true);
    EXPECT_TRUE(req.ReduceStopForBest());

    auto& ref = req.WithReduceStopForBest(false);
    EXPECT_FALSE(req.ReduceStopForBest());
    EXPECT_EQ(&ref, &req);
}
