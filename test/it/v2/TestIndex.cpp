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

#include <memory>

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"

using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::CreateIndexRequest;
using ::milvus::proto::milvus::DescribeIndexRequest;
using ::milvus::proto::milvus::DescribeIndexResponse;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Property;

namespace {

std::shared_ptr<milvus::MilvusClientV2>
CreateConnectedV2Client(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{"127.0.0.1", port};
    auto status = client->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());
    return client;
}

void
AddIndexDescription(DescribeIndexResponse* response, const std::string& field_name, const std::string& index_name) {
    auto* desc = response->add_index_descriptions();
    desc->set_field_name(field_name);
    desc->set_index_name(index_name);
    auto* kv = desc->add_params();
    kv->set_key(milvus::INDEX_TYPE);
    kv->set_value(std::to_string(milvus::IndexType::IVF_FLAT));
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, ListIndexesFiltersByFieldName) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeIndex(_, Property(&DescribeIndexRequest::collection_name, "collection"), _))
        .WillOnce([](::grpc::ServerContext*, const DescribeIndexRequest*, DescribeIndexResponse* response) {
            AddIndexDescription(response, "vec_field_a", "index_a");
            AddIndexDescription(response, "vec_field_b", "index_b");
            return ::grpc::Status{};
        });

    milvus::ListIndexesResponse response;
    auto status = client->ListIndexes(milvus::ListIndexesRequest()
                                          .WithDatabaseName("db")
                                          .WithCollectionName("collection")
                                          .WithFieldName("vec_field_a"),
                                      response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_THAT(response.IndexNames(), ElementsAre("index_a"));
    ASSERT_EQ(response.Descs().size(), 1);
    EXPECT_EQ(response.Descs()[0].FieldName(), "vec_field_a");
}

TEST_F(UnconnectMilvusMockedTest, ListIndexesWithoutFieldNameReturnsAll) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeIndex(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const DescribeIndexRequest*, DescribeIndexResponse* response) {
            AddIndexDescription(response, "vec_field_a", "index_a");
            AddIndexDescription(response, "vec_field_b", "index_b");
            return ::grpc::Status{};
        });

    milvus::ListIndexesResponse response;
    auto status = client->ListIndexes(
        milvus::ListIndexesRequest().WithDatabaseName("db").WithCollectionName("collection"), response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_THAT(response.IndexNames(), ElementsAre("index_a", "index_b"));
}

TEST_F(UnconnectMilvusMockedTest, CreateIndexRejectsEmptyIndexes) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    // StrictMock: any unexpected CreateIndex RPC would fail the test.
    milvus::Status status = client->CreateIndex(milvus::CreateIndexRequest());
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, CreateIndexWithIndexParams) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, CreateIndex(_, Property(&CreateIndexRequest::field_name, "vec_field"), _))
        .WillOnce([](::grpc::ServerContext*, const CreateIndexRequest* request,
                     ::milvus::proto::common::Status* status) {
            EXPECT_EQ(request->collection_name(), "collection");
            EXPECT_EQ(request->index_name(), "vec_idx");
            std::unordered_map<std::string, std::string> params{};
            for (const auto& pair : request->extra_params()) {
                params.emplace(pair.key(), pair.value());
            }
            EXPECT_EQ(params[milvus::INDEX_TYPE], std::to_string(milvus::IndexType::HNSW));
            EXPECT_EQ(params[milvus::METRIC_TYPE], std::to_string(milvus::MetricType::L2));
            EXPECT_EQ(params["params"], R"({"M":"16"})");
            status->set_code(milvus::proto::common::ErrorCode::Success);
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, CreateIndex(_, Property(&CreateIndexRequest::field_name, "text_field"), _))
        .WillOnce([](::grpc::ServerContext*, const CreateIndexRequest* request,
                     ::milvus::proto::common::Status* status) {
            EXPECT_EQ(request->collection_name(), "collection");
            EXPECT_EQ(request->index_name(), "text_idx");
            std::unordered_map<std::string, std::string> params{};
            for (const auto& pair : request->extra_params()) {
                params.emplace(pair.key(), pair.value());
            }
            EXPECT_EQ(params[milvus::INDEX_TYPE], std::to_string(milvus::IndexType::INVERTED));
            // scalar field index has no metric type
            EXPECT_EQ(params.count(milvus::METRIC_TYPE), 0);
            status->set_code(milvus::proto::common::ErrorCode::Success);
            return ::grpc::Status{};
        });

    milvus::IndexParam param_vec("vec_field", "vec_idx", milvus::IndexType::HNSW, milvus::MetricType::L2);
    param_vec.AddExtraParam("M", "16");
    milvus::IndexParam param_scalar("text_field", "text_idx", milvus::IndexType::INVERTED);
    auto status = client->CreateIndex(milvus::CreateIndexRequest()
                                          .WithDatabaseName("db")
                                          .WithCollectionName("collection")
                                          .AddIndexParam(std::move(param_vec))
                                          .AddIndexParam(std::move(param_scalar))
                                          .WithSync(false));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, CreateIndexDeprecatedIndexDescForwarding) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, CreateIndex(_, Property(&CreateIndexRequest::field_name, "vec_field"), _))
        .WillOnce([](::grpc::ServerContext*, const CreateIndexRequest* request,
                     ::milvus::proto::common::Status* status) {
            EXPECT_EQ(request->collection_name(), "collection");
            EXPECT_EQ(request->index_name(), "vec_idx");
            std::unordered_map<std::string, std::string> params{};
            for (const auto& pair : request->extra_params()) {
                params.emplace(pair.key(), pair.value());
            }
            EXPECT_EQ(params[milvus::INDEX_TYPE], std::to_string(milvus::IndexType::HNSW));
            EXPECT_EQ(params[milvus::METRIC_TYPE], std::to_string(milvus::MetricType::L2));
            EXPECT_EQ(params["params"], R"({"M":"16"})");
            status->set_code(milvus::proto::common::ErrorCode::Success);
            return ::grpc::Status{};
        });

    milvus::IndexDesc index_desc("vec_field", "vec_idx", milvus::IndexType::HNSW, milvus::MetricType::L2);
    index_desc.AddExtraParam("M", "16");
    auto status = client->CreateIndex(milvus::CreateIndexRequest()
                                          .WithDatabaseName("db")
                                          .WithCollectionName("collection")
                                          .WithIndexes({std::move(index_desc)})
                                          .WithSync(false));
    EXPECT_TRUE(status.IsOk());
}
