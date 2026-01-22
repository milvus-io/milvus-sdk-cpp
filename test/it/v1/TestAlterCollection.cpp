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
using ::milvus::proto::milvus::AlterCollectionRequest;

using ::testing::_;
using ::testing::AllOf;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

TEST_F(MilvusMockedTest, AlterCollectionProperties) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection_name = "Foo";
    EXPECT_CALL(service_, AlterCollection(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const AlterCollectionRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_NE(request, nullptr);
            EXPECT_EQ(request->collection_name(), collection_name);

            std::unordered_map<std::string, std::string> got;
            for (const auto& kv : request->properties()) {
                got[kv.key()] = kv.value();
            }
            EXPECT_EQ(got[milvus::MMAP_ENABLED], "true");
            EXPECT_EQ(got[milvus::COLLECTION_TTL_SECONDS], "60");
            return ::grpc::Status{};
        });
    std::unordered_map<std::string, std::string> properties{};
    properties[milvus::MMAP_ENABLED] = "true";
    properties[milvus::COLLECTION_TTL_SECONDS] = "60";
    auto status = client_->AlterCollectionProperties(collection_name, properties);
    EXPECT_TRUE(status.IsOk());
}
