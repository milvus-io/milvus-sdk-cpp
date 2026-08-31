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
