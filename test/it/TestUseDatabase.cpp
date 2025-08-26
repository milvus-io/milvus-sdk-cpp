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
using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::testing::_;
using ::testing::Property;

TEST_F(UnconnectMilvusMockedTest, UseDatabase) {
    std::string db_name;
    auto status = client_->CurrentUsedDatabase(db_name);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::NOT_CONNECTED);

    EXPECT_CALL(service_, Connect(_, _, _))
        .WillRepeatedly(
            [](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) { return ::grpc::Status{}; });

    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    connect_param.SetDbName("AAA");
    status = client_->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());

    status = client_->CurrentUsedDatabase(db_name);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(db_name, "AAA");

    status = client_->UseDatabase("BBB");
    EXPECT_TRUE(status.IsOk());

    status = client_->CurrentUsedDatabase(db_name);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(db_name, "BBB");
}
