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

#include "TypeUtils.h"
#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::DescribeCollectionRequest;
using ::milvus::proto::milvus::DescribeCollectionResponse;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, DescribeCollectionFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    milvus::FieldSchema field_schema{"f0", milvus::DataType::DOUBLE, "f0_desc", true, false};

    const int64_t collection_id = 100;
    const int32_t shards_num = 3;
    const std::vector<std::string> aliases = {"a", "b", "c"};
    const int64_t created_ts = 999;

    milvus::CollectionSchema collection_schema("test", "test_desc", 3);
    collection_schema.AddField(field_schema);

    EXPECT_CALL(service_, DescribeCollection(
                              _, Property(&DescribeCollectionRequest::collection_name, collection_schema.Name()), _))
        .WillOnce([&collection_id, &shards_num, &aliases, &created_ts, &collection_schema](
                      ::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
            response->set_collectionid(collection_id);
            response->set_shards_num(shards_num);
            for (auto& alias : aliases) {
                response->add_aliases(alias);
            }
            response->set_created_timestamp(created_ts);
            auto proto_schema = response->mutable_schema();
            milvus::ConvertCollectionSchema(collection_schema, *proto_schema);
            return ::grpc::Status{};
        });

    ::milvus::CollectionDesc desc;
    auto status = client_->DescribeCollection(collection_schema.Name(), desc);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(desc.Schema().Name(), collection_schema.Name());
    EXPECT_EQ(desc.Schema().Description(), collection_schema.Description());
    EXPECT_EQ(desc.Schema().ShardsNum(), shards_num);
    EXPECT_THAT(desc.Alias(), ElementsAreArray(aliases));
    EXPECT_EQ(desc.ID(), collection_id);
    EXPECT_EQ(desc.CreatedTime(), created_ts);
    EXPECT_EQ(desc.Schema().Fields().size(), 1);

    auto field = desc.Schema().Fields().at(0);
    EXPECT_EQ(field.Name(), field_schema.Name());
    EXPECT_EQ(field.Description(), field_schema.Description());
    EXPECT_EQ(field.FieldDataType(), field_schema.FieldDataType());
    EXPECT_EQ(field.IsPrimaryKey(), field_schema.IsPrimaryKey());
    EXPECT_EQ(field.AutoID(), field_schema.AutoID());
}

TEST_F(MilvusMockedTest, DescribeCollectionFooConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    const std::string collection_name = "test";
    ::milvus::CollectionDesc desc;
    auto status = client_->DescribeCollection(collection_name, desc);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, DescribeCollectionFooFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection_name = "test";

    EXPECT_CALL(service_,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillOnce([](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
        });

    ::milvus::CollectionDesc desc;
    auto status = client_->DescribeCollection(collection_name, desc);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}