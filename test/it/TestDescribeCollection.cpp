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

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::DescribeCollectionRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, DescribeCollectionFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_schema = "Foo";
    std::string collection_name = "Bar";
    milvus::proto::milvus::DescribeCollectionRequest request;
    request.set_collection_name(collection_name);
    request.set_collectionid(1);
    milvus::proto::milvus::DescribeCollectionResponse response;

    auto status = client_->DescribeCollection(request, response);
    EXPECT_TRUE(status.IsOk());
}