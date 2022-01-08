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

TEST_F(MilvusServerTest, CreateAndDeleteCollection) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    milvus::CollectionSchema collection_schema("Foo");
    collection_schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "id", true, true));
    collection_schema.AddField(milvus::FieldSchema("name", milvus::DataType::STRING, "name"));
    collection_schema.AddField(milvus::FieldSchema("age", milvus::DataType::INT16, "age"));
    collection_schema.AddField(
        milvus::FieldSchema("face", milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(1024));

    auto status = client_->CreateCollection(collection_schema);
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_TRUE(status.IsOk());

    status = client_->DropCollection("Foo");
    EXPECT_TRUE(status.IsOk());
}