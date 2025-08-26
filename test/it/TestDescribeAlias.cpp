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

#include "milvus/types/AliasDesc.h"
#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::DescribeAliasRequest;
using ::milvus::proto::milvus::DescribeAliasResponse;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, DescribeAlias) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string db_name = "db";
    const std::string collection_name = "test";
    const std::string alias_name = "alias";

    EXPECT_CALL(service_, DescribeAlias(_, AllOf(Property(&DescribeAliasRequest::alias, alias_name)), _))
        .WillOnce([&](::grpc::ServerContext*, const DescribeAliasRequest* request, DescribeAliasResponse* response) {
            response->set_db_name(db_name);
            response->set_collection(collection_name);
            response->set_alias(alias_name);
            return ::grpc::Status{};
        });

    milvus::AliasDesc desc;
    auto status = client_->DescribeAlias(alias_name, desc);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(desc.DatabaseName(), db_name);
    EXPECT_EQ(desc.CollectionName(), collection_name);
    EXPECT_EQ(desc.Name(), alias_name);
}
