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
using ::milvus::proto::milvus::HasCollectionRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, HasCollectionFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_schema = "Foo";

    for (auto value : {true, false}) {
        EXPECT_CALL(service_, HasCollection(_, Property(&HasCollectionRequest::collection_name, collection_schema), _))
            .WillOnce([value](::grpc::ServerContext*, const HasCollectionRequest*,
                              ::milvus::proto::milvus::BoolResponse* response) {
                response->set_value(value);
                return ::grpc::Status{};
            });
        bool has_collection{false};
        auto status = client_->HasCollection(collection_schema, has_collection);

        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(has_collection, value);
    }
}
