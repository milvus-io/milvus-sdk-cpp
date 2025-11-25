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

#include "../mocks/MilvusMockedTest.h"
#include "milvus/types/Constants.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::AlterIndexRequest;

using ::testing::_;
using ::testing::AllOf;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

TEST_F(MilvusMockedTest, AlterIndexProperties) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection_name = "Foo";
    const std::string index_name = "Bar";
    EXPECT_CALL(service_,
                AlterIndex(_,
                           AllOf(Property(&AlterIndexRequest::collection_name, collection_name),
                                 Property(&AlterIndexRequest::index_name, index_name),
                                 Property(&AlterIndexRequest::extra_params,
                                          UnorderedElementsAre(milvus::TestKv(milvus::MMAP_ENABLED, "true")))),
                           _))
        .WillOnce([](::grpc::ServerContext*, const AlterIndexRequest* request, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });
    std::unordered_map<std::string, std::string> properties{};
    properties[milvus::MMAP_ENABLED] = "true";
    auto status = client_->AlterIndexProperties(collection_name, index_name, properties);
    EXPECT_TRUE(status.IsOk());
}
