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
#include "milvus/types/AliasDesc.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::ListAliasesRequest;
using ::milvus::proto::milvus::ListAliasesResponse;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, ListAliases) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string db_name = "db";
    const std::string collection_name = "test";
    const std::vector<std::string> alias_names = {"a", "b", "c"};

    EXPECT_CALL(service_, ListAliases(_, AllOf(Property(&ListAliasesRequest::collection_name, collection_name)), _))
        .WillOnce([&](::grpc::ServerContext*, const ListAliasesRequest* request, ListAliasesResponse* response) {
            response->set_db_name(db_name);
            response->set_collection_name(collection_name);
            for (auto& name : alias_names) {
                response->add_aliases()->append(name);
            }
            return ::grpc::Status{};
        });

    std::vector<milvus::AliasDesc> descs;
    auto status = client_->ListAliases(collection_name, descs);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(alias_names.size(), descs.size());
    for (auto i = 0; i < alias_names.size(); i++) {
        auto& desc = descs[i];
        EXPECT_EQ(desc.DatabaseName(), db_name);
        EXPECT_EQ(desc.CollectionName(), collection_name);
        EXPECT_EQ(desc.Name(), alias_names[i]);
    }
}
