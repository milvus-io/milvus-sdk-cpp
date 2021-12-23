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
using ::milvus::proto::milvus::HasPartitionRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, HasPartitionFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    const std::string partition{"Bar"};

    for (auto value : {true, false}) {
        EXPECT_CALL(service_, HasPartition(_,
                                           AllOf(Property(&HasPartitionRequest::collection_name, collection),

                                                 Property(&HasPartitionRequest::partition_name, partition)),
                                           _))
            .WillOnce([value](::grpc::ServerContext*, const HasPartitionRequest*,
                              ::milvus::proto::milvus::BoolResponse* response) {
                response->set_value(value);
                return ::grpc::Status{};
            });
        bool has{false};
        auto status = client_->HasPartition(collection, partition, has);

        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(has, value);
    }
}
