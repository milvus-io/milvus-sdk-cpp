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

using testing::UnorderedElementsAre;
using testing::UnorderedElementsAreArray;

class MilvusServerTestSearch : public MilvusServerTest {
 protected:
    void
    SetUp() override {
        MilvusServerTest::SetUp();
        milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
        client_->Connect(connect_param);
    }

    void
    TearDown() override {
        MilvusServerTest::TearDown();
    }
};

TEST_F(MilvusServerTestSearch, SearchWithoutIndex) {
    milvus::CollectionSchema collection_schema("Foo");
    collection_schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
    collection_schema.AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
    collection_schema.AddField(
        milvus::FieldSchema("face", milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(4));

    auto status = client_->CreateCollection(collection_schema);
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_TRUE(status.IsOk());

    status = client_->CreatePartition("Foo", "Bar");
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_TRUE(status.IsOk());

    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::Int16FieldData>("age", std::vector<int16_t>{12, 13}),
        std::make_shared<milvus::FloatVecFieldData>(
            "face", std::vector<std::vector<float>>{std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f},
                                                    std::vector<float>{0.5f, 0.6f, 0.7f, 0.8f}})};
    milvus::IDArray id_array{std::vector<int64_t>{}};
    status = client_->Insert("Foo", "Bar", fields, id_array);
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(id_array.IntIDArray().size(), 2);

    milvus::ProgressMonitor progress_monitor{5};
    status = client_->LoadPartitions("Foo", std::vector<std::string>{"Bar"}, progress_monitor);
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_TRUE(status.IsOk());

    milvus::SearchArguments arguments{};
    arguments.SetCollectionName("Foo");
    arguments.AddPartitionName("Bar");
    arguments.SetTopK(10);
    arguments.AddOutputField("age");
    arguments.SetExpression("id > 0");
    arguments.AddTargetVector("face", std::vector<float>{0.f, 0.f, 0.f, 0.f});
    arguments.AddTargetVector("face", std::vector<float>{1.f, 1.f, 1.f, 1.f});
    milvus::SearchResults search_results{};
    status = client_->Search(arguments, search_results);
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_TRUE(status.IsOk());

    const auto& results = search_results.Results();
    EXPECT_EQ(results.size(), 2);
    EXPECT_THAT(results.at(0).Ids().IntIDArray(), UnorderedElementsAreArray(id_array.IntIDArray()));
    EXPECT_THAT(results.at(1).Ids().IntIDArray(), UnorderedElementsAreArray(id_array.IntIDArray()));

    EXPECT_EQ(results.at(0).Scores().size(), 2);
    EXPECT_EQ(results.at(1).Scores().size(), 2);

    EXPECT_LT(results.at(0).Scores().at(0), results.at(0).Scores().at(1));
    EXPECT_LT(results.at(1).Scores().at(0), results.at(1).Scores().at(1));

    // match fields: age
    EXPECT_EQ(results.at(0).OutputFields().size(), 1);
    EXPECT_EQ(results.at(1).OutputFields().size(), 1);
    EXPECT_THAT(dynamic_cast<milvus::Int16FieldData&>(*results.at(0).OutputField("age")).Data(),
                UnorderedElementsAre(12, 13));
    EXPECT_THAT(dynamic_cast<milvus::Int16FieldData&>(*results.at(1).OutputField("age")).Data(),
                UnorderedElementsAre(12, 13));

    status = client_->DropCollection("Foo");
    EXPECT_TRUE(status.IsOk());
}