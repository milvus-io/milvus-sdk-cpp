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

class CreateCollectionRequestTest : public ::testing::Test {};

TEST_F(CreateCollectionRequestTest, GettersAndSetters) {
    milvus::CreateCollectionRequest req;

    // CollectionName
    req.WithCollectionName("test_coll");
    EXPECT_EQ(req.CollectionName(), "test_coll");

    // DatabaseName
    req.WithDatabaseName("test_db");
    EXPECT_EQ(req.DatabaseName(), "test_db");

    // Description
    req.WithDescription("my description");
    EXPECT_EQ(req.Description(), "my description");

    // CollectionSchema
    auto schema = std::make_shared<milvus::CollectionSchema>("test_coll");
    req.WithCollectionSchema(schema);
    EXPECT_EQ(req.CollectionSchema()->Name(), "test_coll");

    // NumPartitions
    req.WithNumPartitions(16);
    EXPECT_EQ(req.NumPartitions(), 16);

    // NumShards
    req.WithNumShards(4);
    EXPECT_EQ(req.NumShards(), 4);

    // ConsistencyLevel
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    EXPECT_EQ(req.GetConsistencyLevel(), milvus::ConsistencyLevel::STRONG);

    // Properties
    std::unordered_map<std::string, std::string> props;
    props["key1"] = "val1";
    req.WithProperties(std::move(props));
    EXPECT_EQ(req.Properties().at("key1"), "val1");

    // AddProperty
    req.AddProperty("key2", "val2");
    EXPECT_EQ(req.Properties().at("key2"), "val2");

    // Indexes
    milvus::IndexDesc idx;
    std::vector<milvus::IndexDesc> indexes;
    indexes.push_back(idx);
    req.WithIndexes(std::move(indexes));
    EXPECT_EQ(req.Indexes().size(), 1);

    // AddIndex
    milvus::IndexDesc idx2;
    req.AddIndex(std::move(idx2));
    EXPECT_EQ(req.Indexes().size(), 2);
}

TEST_F(CreateCollectionRequestTest, FluentChaining) {
    milvus::CreateCollectionRequest req;
    auto& ref = req.WithCollectionName("c1")
                    .WithDatabaseName("db1")
                    .WithDescription("desc")
                    .WithNumPartitions(8)
                    .WithNumShards(2)
                    .WithConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);
    EXPECT_EQ(&ref, &req);
    EXPECT_EQ(req.CollectionName(), "c1");
    EXPECT_EQ(req.DatabaseName(), "db1");
}

class DropCollectionRequestTest : public ::testing::Test {};

TEST_F(DropCollectionRequestTest, GettersAndSetters) {
    milvus::DropCollectionRequest req;
    req.WithCollectionName("drop_coll");
    EXPECT_EQ(req.CollectionName(), "drop_coll");

    req.WithDatabaseName("drop_db");
    EXPECT_EQ(req.DatabaseName(), "drop_db");
}

class HasCollectionRequestTest : public ::testing::Test {};

TEST_F(HasCollectionRequestTest, GettersAndSetters) {
    milvus::HasCollectionRequest req;
    req.WithCollectionName("has_coll");
    EXPECT_EQ(req.CollectionName(), "has_coll");
}

class DescribeCollectionRequestTest : public ::testing::Test {};

TEST_F(DescribeCollectionRequestTest, GettersAndSetters) {
    milvus::DescribeCollectionRequest req;
    req.WithCollectionName("desc_coll");
    EXPECT_EQ(req.CollectionName(), "desc_coll");
}

class LoadCollectionRequestTest : public ::testing::Test {};

TEST_F(LoadCollectionRequestTest, GettersAndSetters) {
    milvus::LoadCollectionRequest req;

    req.WithCollectionName("load_coll");
    EXPECT_EQ(req.CollectionName(), "load_coll");

    // Sync
    req.WithSync(false);
    EXPECT_FALSE(req.Sync());
    req.WithSync(true);
    EXPECT_TRUE(req.Sync());

    // ReplicaNum
    req.WithReplicaNum(3);
    EXPECT_EQ(req.ReplicaNum(), 3);

    // TimeoutMs
    req.WithTimeoutMs(30000);
    EXPECT_EQ(req.TimeoutMs(), 30000);

    // Refresh
    req.WithRefresh(true);
    EXPECT_TRUE(req.Refresh());

    // LoadFields
    std::set<std::string> fields{"f1", "f2"};
    req.WithLoadFields(fields);
    EXPECT_EQ(req.LoadFields().size(), 2);
    EXPECT_TRUE(req.LoadFields().count("f1"));

    // AddLoadField
    req.AddLoadField("f3");
    EXPECT_TRUE(req.LoadFields().count("f3"));

    // SkipDynamicField
    req.WithSkipDynamicField(true);
    EXPECT_TRUE(req.SkipDynamicField());

    // TargetResourceGroups
    std::set<std::string> groups{"rg1", "rg2"};
    req.WithTargetResourceGroups(groups);
    EXPECT_EQ(req.TargetResourceGroups().size(), 2);
    EXPECT_TRUE(req.TargetResourceGroups().count("rg1"));
}

class RefreshLoadRequestTest : public ::testing::Test {};

TEST_F(RefreshLoadRequestTest, GettersAndSetters) {
    milvus::RefreshLoadRequest req;
    EXPECT_TRUE(req.Sync());
    EXPECT_EQ(req.TimeoutMs(), 60000);

    auto& ref = req.WithDatabaseName("db").WithCollectionName("refresh_coll").WithSync(false).WithTimeoutMs(30000);
    EXPECT_EQ(&ref, &req);
    EXPECT_EQ(req.DatabaseName(), "db");
    EXPECT_EQ(req.CollectionName(), "refresh_coll");
    EXPECT_FALSE(req.Sync());
    EXPECT_EQ(req.TimeoutMs(), 30000);
}

class ReleaseCollectionRequestTest : public ::testing::Test {};

TEST_F(ReleaseCollectionRequestTest, GettersAndSetters) {
    milvus::ReleaseCollectionRequest req;
    req.WithCollectionName("release_coll");
    EXPECT_EQ(req.CollectionName(), "release_coll");
}

class RenameCollectionRequestTest : public ::testing::Test {};

TEST_F(RenameCollectionRequestTest, GettersAndSetters) {
    milvus::RenameCollectionRequest req;
    req.WithCollectionName("old_name");
    EXPECT_EQ(req.CollectionName(), "old_name");

    req.WithNewCollectionName("new_name");
    EXPECT_EQ(req.NewCollectionName(), "new_name");
}

class TruncateCollectionRequestTest : public ::testing::Test {};

TEST_F(TruncateCollectionRequestTest, GettersAndSetters) {
    milvus::TruncateCollectionRequest req;
    req.WithCollectionName("trunc_coll");
    EXPECT_EQ(req.CollectionName(), "trunc_coll");
}

class BatchDescribeCollectionsRequestTest : public ::testing::Test {};

TEST_F(BatchDescribeCollectionsRequestTest, GettersAndSetters) {
    milvus::BatchDescribeCollectionsRequest req;

    auto& db_ref = req.WithDatabaseName("test_db");
    EXPECT_EQ(&db_ref, &req);
    EXPECT_EQ(req.DatabaseName(), "test_db");

    std::vector<std::string> names{"coll1", "coll2"};
    auto& names_ref = req.WithCollectionNames(std::move(names));
    EXPECT_EQ(&names_ref, &req);
    ASSERT_EQ(req.CollectionNames().size(), 2);
    EXPECT_EQ(req.CollectionNames()[0], "coll1");

    auto& add_name_ref = req.AddCollectionName("coll3");
    EXPECT_EQ(&add_name_ref, &req);
    ASSERT_EQ(req.CollectionNames().size(), 3);
    EXPECT_EQ(req.CollectionNames()[2], "coll3");

    std::vector<int64_t> ids{101, 102};
    auto& ids_ref = req.WithCollectionIDs(std::move(ids));
    EXPECT_EQ(&ids_ref, &req);
    ASSERT_EQ(req.CollectionIDs().size(), 2);
    EXPECT_EQ(req.CollectionIDs()[0], 101);

    auto& add_id_ref = req.AddCollectionID(103);
    EXPECT_EQ(&add_id_ref, &req);
    ASSERT_EQ(req.CollectionIDs().size(), 3);
    EXPECT_EQ(req.CollectionIDs()[2], 103);
}

class DescribeReplicasRequestTest : public ::testing::Test {};

TEST_F(DescribeReplicasRequestTest, GettersAndSetters) {
    milvus::DescribeReplicasRequest req;
    auto& ref = req.WithDatabaseName("test_db").WithCollectionName("test_coll");
    EXPECT_EQ(&ref, &req);
    EXPECT_EQ(req.DatabaseName(), "test_db");
    EXPECT_EQ(req.CollectionName(), "test_coll");
}

class ListCollectionsRequestTest : public ::testing::Test {};

TEST_F(ListCollectionsRequestTest, GettersAndSetters) {
    milvus::ListCollectionsRequest req;

    req.WithDatabaseName("test_db");
    EXPECT_EQ(req.DatabaseName(), "test_db");

    req.WithOnlyShowLoaded(true);
    EXPECT_TRUE(req.OnlyShowLoaded());

    req.WithOnlyShowLoaded(false);
    EXPECT_FALSE(req.OnlyShowLoaded());
}

class GetCollectionStatsRequestTest : public ::testing::Test {};

TEST_F(GetCollectionStatsRequestTest, GettersAndSetters) {
    milvus::GetCollectionStatsRequest req;
    req.WithCollectionName("stats_coll");
    EXPECT_EQ(req.CollectionName(), "stats_coll");
}

class GetLoadStateRequestTest : public ::testing::Test {};

TEST_F(GetLoadStateRequestTest, GettersAndSetters) {
    milvus::GetLoadStateRequest req;
    req.WithCollectionName("load_state_coll");
    EXPECT_EQ(req.CollectionName(), "load_state_coll");

    req.AddPartitionName("p1");
    req.AddPartitionName("p2");
    EXPECT_EQ(req.PartitionNames().size(), 2);
    EXPECT_TRUE(req.PartitionNames().count("p1"));
    EXPECT_TRUE(req.PartitionNames().count("p2"));
}

TEST_F(GetLoadStateRequestTest, SetPartitionNames) {
    milvus::GetLoadStateRequest req;
    std::set<std::string> names = {"p1", "p2"};
    req.SetPartitionNames(std::move(names));
    EXPECT_EQ(req.PartitionNames().size(), 2);
    EXPECT_TRUE(req.PartitionNames().count("p1"));

    std::set<std::string> names2 = {"p3"};
    auto& ref = req.WithPartitionNames(std::move(names2));
    EXPECT_EQ(req.PartitionNames().size(), 1);
    EXPECT_TRUE(req.PartitionNames().count("p3"));
    EXPECT_EQ(&ref, &req);
}

class AddCollectionFieldRequestTest : public ::testing::Test {};

TEST_F(AddCollectionFieldRequestTest, GettersAndSetters) {
    milvus::AddCollectionFieldRequest req;
    req.WithCollectionName("add_field_coll");
    EXPECT_EQ(req.CollectionName(), "add_field_coll");

    milvus::FieldSchema field;
    field.SetName("my_field");
    req.WithField(std::move(field));
    EXPECT_EQ(req.Field().Name(), "my_field");
}

class AddCollectionStructFieldRequestTest : public ::testing::Test {};

TEST_F(AddCollectionStructFieldRequestTest, GettersAndSetters) {
    milvus::AddCollectionStructFieldRequest req;
    req.WithCollectionName("add_struct_field_coll");
    EXPECT_EQ(req.CollectionName(), "add_struct_field_coll");

    milvus::StructFieldSchema field;
    field.WithName("my_struct_field")
        .WithMaxCapacity(8)
        .AddField(milvus::FieldSchema("sub_int", milvus::DataType::INT32))
        .AddField(milvus::FieldSchema("sub_text", milvus::DataType::VARCHAR).WithMaxLength(64));
    req.WithStructField(std::move(field));
    EXPECT_EQ(req.StructField().Name(), "my_struct_field");
    EXPECT_EQ(req.StructField().Fields().size(), 2);
}

class DropCollectionFieldRequestTest : public ::testing::Test {};

TEST_F(DropCollectionFieldRequestTest, GettersAndSetters) {
    milvus::DropCollectionFieldRequest req;
    req.WithCollectionName("drop_field_coll").WithDatabaseName("my_db");
    EXPECT_EQ(req.CollectionName(), "drop_field_coll");
    EXPECT_EQ(req.DatabaseName(), "my_db");
    EXPECT_EQ(req.FieldName(), "");
    EXPECT_EQ(req.FieldID(), 0);

    auto& ref = req.WithFieldName("my_field");
    EXPECT_EQ(&ref, &req);
    EXPECT_EQ(req.FieldName(), "my_field");
    EXPECT_EQ(req.FieldID(), 0);
}

TEST_F(DropCollectionFieldRequestTest, Setters) {
    milvus::DropCollectionFieldRequest req;
    req.SetFieldID(202);
    EXPECT_EQ(req.FieldName(), "");
    EXPECT_EQ(req.FieldID(), 202);
}

class AddFunctionFieldRequestTest : public ::testing::Test {};

TEST_F(AddFunctionFieldRequestTest, GettersAndSetters) {
    milvus::AddFunctionFieldRequest req;
    req.WithCollectionName("add_fn_field_coll").WithDatabaseName("my_db");
    EXPECT_EQ(req.CollectionName(), "add_fn_field_coll");
    EXPECT_EQ(req.DatabaseName(), "my_db");
    EXPECT_EQ(req.Function(), nullptr);

    milvus::FieldSchema field;
    field.SetName("sparse_vec");
    field.SetDataType(milvus::DataType::SPARSE_FLOAT_VECTOR);

    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25, "tokenize");
    function->AddInputFieldName("text");
    function->AddOutputFieldName("sparse_vec");
    milvus::IndexDesc index("sparse_vec", "sparse_idx", milvus::IndexType::SPARSE_INVERTED_INDEX,
                            milvus::MetricType::BM25);
    index.AddExtraParam("drop_ratio_build", "0.2");

    auto& ref = req.WithField(std::move(field)).WithFunction(function).WithIndex(std::move(index));
    EXPECT_EQ(&ref, &req);
    EXPECT_EQ(req.Field().Name(), "sparse_vec");
    ASSERT_NE(req.Function(), nullptr);
    EXPECT_EQ(req.Function()->Name(), "bm25_fn");
    EXPECT_EQ(req.Function()->OutputFieldNames().size(), 1);
    EXPECT_EQ(req.Function()->OutputFieldNames()[0], "sparse_vec");
    EXPECT_EQ(req.Index().FieldName(), "sparse_vec");
    EXPECT_EQ(req.Index().IndexName(), "sparse_idx");
    EXPECT_EQ(req.Index().IndexType(), milvus::IndexType::SPARSE_INVERTED_INDEX);
    EXPECT_EQ(req.Index().MetricType(), milvus::MetricType::BM25);
    EXPECT_EQ(req.Index().ExtraParams().at("drop_ratio_build"), "0.2");
}

TEST_F(AddFunctionFieldRequestTest, SetFunction) {
    milvus::AddFunctionFieldRequest req;
    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
    req.SetFunction(function);
    ASSERT_NE(req.Function(), nullptr);
    EXPECT_EQ(req.Function()->Name(), "bm25_fn");
}

TEST_F(AddFunctionFieldRequestTest, SetIndex) {
    milvus::AddFunctionFieldRequest req;
    milvus::IndexDesc index("sparse_vec", "sparse_idx", milvus::IndexType::SPARSE_WAND, milvus::MetricType::BM25);
    req.SetIndex(std::move(index));
    EXPECT_EQ(req.Index().FieldName(), "sparse_vec");
    EXPECT_EQ(req.Index().IndexName(), "sparse_idx");
    EXPECT_EQ(req.Index().IndexType(), milvus::IndexType::SPARSE_WAND);
}

class DropFunctionFieldRequestTest : public ::testing::Test {};

TEST_F(DropFunctionFieldRequestTest, GettersAndSetters) {
    milvus::DropFunctionFieldRequest req;
    req.WithCollectionName("drop_fn_field_coll").WithDatabaseName("my_db");
    EXPECT_EQ(req.CollectionName(), "drop_fn_field_coll");
    EXPECT_EQ(req.DatabaseName(), "my_db");
    EXPECT_EQ(req.FunctionName(), "");

    auto& ref = req.WithFunctionName("bm25_fn");
    EXPECT_EQ(&ref, &req);
    EXPECT_EQ(req.FunctionName(), "bm25_fn");
}

TEST_F(DropFunctionFieldRequestTest, SetFunctionName) {
    milvus::DropFunctionFieldRequest req;
    req.SetFunctionName("another_fn");
    EXPECT_EQ(req.FunctionName(), "another_fn");
}

class AddCollectionFunctionRequestTest : public ::testing::Test {};
TEST_F(AddCollectionFunctionRequestTest, GettersAndSetters) {
    milvus::AddCollectionFunctionRequest req;

    // inherited: collection name and database name
    req.WithCollectionName("add_fn_coll").WithDatabaseName("my_db");
    EXPECT_EQ(req.CollectionName(), "add_fn_coll");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    // default function is null
    EXPECT_EQ(req.Function(), nullptr);

    auto function = std::make_shared<milvus::Function>("my_fn", milvus::FunctionType::BM25, "tokenize");
    function->AddInputFieldName("text");
    function->AddOutputFieldName("sparse_vec");

    auto& ref = req.WithFunction(function);
    EXPECT_EQ(&ref, &req);
    ASSERT_NE(req.Function(), nullptr);
    EXPECT_EQ(req.Function()->Name(), "my_fn");
    EXPECT_EQ(req.Function()->GetFunctionType(), milvus::FunctionType::BM25);
    EXPECT_EQ(req.Function()->InputFieldNames().size(), 1);
    EXPECT_EQ(req.Function()->InputFieldNames()[0], "text");
    EXPECT_EQ(req.Function()->OutputFieldNames().size(), 1);
    EXPECT_EQ(req.Function()->OutputFieldNames()[0], "sparse_vec");
}

TEST_F(AddCollectionFunctionRequestTest, SetFunction) {
    milvus::AddCollectionFunctionRequest req;
    auto function = std::make_shared<milvus::Function>("my_fn", milvus::FunctionType::BM25);
    req.SetFunction(function);
    ASSERT_NE(req.Function(), nullptr);
    EXPECT_EQ(req.Function()->Name(), "my_fn");
}

class AlterCollectionFunctionRequestTest : public ::testing::Test {};

TEST_F(AlterCollectionFunctionRequestTest, GettersAndSetters) {
    milvus::AlterCollectionFunctionRequest req;
    req.WithCollectionName("alter_fn_coll");
    EXPECT_EQ(req.CollectionName(), "alter_fn_coll");

    EXPECT_EQ(req.Function(), nullptr);

    auto function = std::make_shared<milvus::Function>("my_fn", milvus::FunctionType::BM25);
    auto& ref = req.WithFunction(function);
    EXPECT_EQ(&ref, &req);
    ASSERT_NE(req.Function(), nullptr);
    EXPECT_EQ(req.Function()->Name(), "my_fn");
}

TEST_F(AlterCollectionFunctionRequestTest, SetFunction) {
    milvus::AlterCollectionFunctionRequest req;
    auto function = std::make_shared<milvus::Function>("my_fn", milvus::FunctionType::BM25);
    req.SetFunction(function);
    ASSERT_NE(req.Function(), nullptr);
    EXPECT_EQ(req.Function()->Name(), "my_fn");
}

class DropCollectionFunctionRequestTest : public ::testing::Test {};

TEST_F(DropCollectionFunctionRequestTest, GettersAndSetters) {
    milvus::DropCollectionFunctionRequest req;
    req.WithCollectionName("drop_fn_coll").WithDatabaseName("my_db");
    EXPECT_EQ(req.CollectionName(), "drop_fn_coll");
    EXPECT_EQ(req.DatabaseName(), "my_db");

    EXPECT_EQ(req.FunctionName(), "");

    auto& ref = req.WithFunctionName("my_fn");
    EXPECT_EQ(&ref, &req);
    EXPECT_EQ(req.FunctionName(), "my_fn");
}

TEST_F(DropCollectionFunctionRequestTest, SetFunctionName) {
    milvus::DropCollectionFunctionRequest req;
    req.SetFunctionName("another_fn");
    EXPECT_EQ(req.FunctionName(), "another_fn");
}

class AlterCollectionPropertiesRequestTest : public ::testing::Test {};

TEST_F(AlterCollectionPropertiesRequestTest, GettersAndSetters) {
    milvus::AlterCollectionPropertiesRequest req;
    req.WithCollectionName("alter_prop_coll");
    EXPECT_EQ(req.CollectionName(), "alter_prop_coll");

    req.AddProperty("k1", "v1");
    EXPECT_EQ(req.Properties().at("k1"), "v1");

    req.AddProperty("k2", "v2");
    EXPECT_EQ(req.Properties().size(), 2);
}

TEST_F(AlterCollectionPropertiesRequestTest, SetProperties) {
    milvus::AlterCollectionPropertiesRequest req;
    std::unordered_map<std::string, std::string> props = {{"k1", "v1"}, {"k2", "v2"}};
    req.SetProperties(std::move(props));
    EXPECT_EQ(req.Properties().size(), 2);
    EXPECT_EQ(req.Properties().at("k1"), "v1");

    std::unordered_map<std::string, std::string> props2 = {{"k3", "v3"}};
    auto& ref = req.WithProperties(std::move(props2));
    EXPECT_EQ(req.Properties().size(), 1);
    EXPECT_EQ(req.Properties().at("k3"), "v3");
    EXPECT_EQ(&ref, &req);
}

class DropCollectionPropertiesRequestTest : public ::testing::Test {};

TEST_F(DropCollectionPropertiesRequestTest, GettersAndSetters) {
    milvus::DropCollectionPropertiesRequest req;
    req.WithCollectionName("drop_prop_coll");
    EXPECT_EQ(req.CollectionName(), "drop_prop_coll");

    req.AddPropertyKey("k1");
    req.AddPropertyKey("k2");
    EXPECT_EQ(req.PropertyKeys().size(), 2);
    EXPECT_TRUE(req.PropertyKeys().count("k1"));
}

TEST_F(DropCollectionPropertiesRequestTest, SetPropertyKeys) {
    milvus::DropCollectionPropertiesRequest req;
    std::set<std::string> keys = {"k1", "k2"};
    req.SetPropertyKeys(std::move(keys));
    EXPECT_EQ(req.PropertyKeys().size(), 2);
    EXPECT_TRUE(req.PropertyKeys().count("k1"));

    std::set<std::string> keys2 = {"k3"};
    auto& ref = req.WithPropertyKeys(std::move(keys2));
    EXPECT_EQ(req.PropertyKeys().size(), 1);
    EXPECT_TRUE(req.PropertyKeys().count("k3"));
    EXPECT_EQ(&ref, &req);
}

class AlterCollectionFieldPropertiesRequestTest : public ::testing::Test {};

TEST_F(AlterCollectionFieldPropertiesRequestTest, GettersAndSetters) {
    milvus::AlterCollectionFieldPropertiesRequest req;
    req.WithCollectionName("alter_field_coll");
    EXPECT_EQ(req.CollectionName(), "alter_field_coll");

    req.WithFieldName("my_field");
    EXPECT_EQ(req.FieldName(), "my_field");

    req.AddProperty("pk1", "pv1");
    EXPECT_EQ(req.Properties().at("pk1"), "pv1");
}

class DropCollectionFieldPropertiesRequestTest : public ::testing::Test {};

TEST_F(DropCollectionFieldPropertiesRequestTest, GettersAndSetters) {
    milvus::DropCollectionFieldPropertiesRequest req;
    req.WithCollectionName("drop_field_prop_coll");
    EXPECT_EQ(req.CollectionName(), "drop_field_prop_coll");

    req.WithFieldName("my_field");
    EXPECT_EQ(req.FieldName(), "my_field");

    req.AddPropertyKey("fk1");
    EXPECT_EQ(req.PropertyKeys().size(), 1);
    EXPECT_TRUE(req.PropertyKeys().count("fk1"));
}

class CreateSimpleCollectionRequestTest : public ::testing::Test {};

TEST_F(CreateSimpleCollectionRequestTest, PrimaryFieldName) {
    milvus::CreateSimpleCollectionRequest req;

    // Default is "id"
    EXPECT_EQ(req.PrimaryFieldName(), "id");

    auto& ref = req.WithPrimaryFieldName("my_pk");
    EXPECT_EQ(req.PrimaryFieldName(), "my_pk");
    EXPECT_EQ(&ref, &req);
}

TEST_F(CreateSimpleCollectionRequestTest, PrimaryFieldType) {
    milvus::CreateSimpleCollectionRequest req;

    // Default is INT64
    EXPECT_EQ(req.PrimaryFieldType(), milvus::DataType::INT64);

    auto& ref = req.WithPrimaryFieldType(milvus::DataType::VARCHAR);
    EXPECT_EQ(req.PrimaryFieldType(), milvus::DataType::VARCHAR);
    EXPECT_EQ(&ref, &req);
}

TEST_F(CreateSimpleCollectionRequestTest, VectorFieldName) {
    milvus::CreateSimpleCollectionRequest req;

    // Default is "vector"
    EXPECT_EQ(req.VectorFieldName(), "vector");

    auto& ref = req.WithVectorFieldName("embedding");
    EXPECT_EQ(req.VectorFieldName(), "embedding");
    EXPECT_EQ(&ref, &req);
}

TEST_F(CreateSimpleCollectionRequestTest, MetricType) {
    milvus::CreateSimpleCollectionRequest req;

    // Default is COSINE
    EXPECT_EQ(req.MetricType(), milvus::MetricType::COSINE);

    auto& ref = req.WithMetricType(milvus::MetricType::L2);
    EXPECT_EQ(req.MetricType(), milvus::MetricType::L2);
    EXPECT_EQ(&ref, &req);
}

TEST_F(CreateSimpleCollectionRequestTest, FluentChaining) {
    milvus::CreateSimpleCollectionRequest req;
    auto& ref = req.WithCollectionName("simple_coll")
                    .WithPrimaryFieldName("pk")
                    .WithPrimaryFieldType(milvus::DataType::VARCHAR)
                    .WithVectorFieldName("vec")
                    .WithDimension(128)
                    .WithMetricType(milvus::MetricType::IP)
                    .WithAutoID(true)
                    .WithEnableDynamicField(false)
                    .WithMaxLength(256)
                    .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    EXPECT_EQ(&ref, &req);
    EXPECT_EQ(req.CollectionName(), "simple_coll");
    EXPECT_EQ(req.PrimaryFieldName(), "pk");
    EXPECT_EQ(req.PrimaryFieldType(), milvus::DataType::VARCHAR);
    EXPECT_EQ(req.VectorFieldName(), "vec");
    EXPECT_EQ(req.Dimension(), 128);
    EXPECT_EQ(req.MetricType(), milvus::MetricType::IP);
    EXPECT_TRUE(req.AutoID());
    EXPECT_FALSE(req.EnableDynamicField());
    EXPECT_EQ(req.MaxLength(), 256);
    EXPECT_EQ(req.ConsistencyLevel(), milvus::ConsistencyLevel::STRONG);
}
