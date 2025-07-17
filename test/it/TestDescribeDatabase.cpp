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
#include "utils/CompareUtils.h"
#include "utils/TypeUtils.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::DescribeDatabaseRequest;
using ::milvus::proto::milvus::DescribeDatabaseResponse;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, DescribeDatabaseSuccess) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string db_name = "test";
    EXPECT_CALL(service_, DescribeDatabase(_, Property(&DescribeDatabaseRequest::db_name, db_name), _))
        .WillOnce([&](::grpc::ServerContext*, const DescribeDatabaseRequest*, DescribeDatabaseResponse* response) {
            response->set_dbid(99);
            response->set_db_name(db_name);
            response->set_created_timestamp(888);
            auto pair = response->add_properties();
            pair->set_key("replicas");
            pair->set_value("2");
            return ::grpc::Status{};
        });

    milvus::DatabaseDesc desc;
    auto status = client_->DescribeDatabase(db_name, desc);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(desc.ID(), 99);
    EXPECT_EQ(desc.Name(), db_name);
    EXPECT_EQ(desc.CreatedTime(), 888);
    auto props = desc.Properties();
    EXPECT_EQ(props.size(), 1);
    EXPECT_TRUE(props.find("replicas") != props.end());
    EXPECT_EQ(props.at("replicas"), "2");
}

TEST_F(MilvusMockedTest, DescribeDatabaseWithoutConnect) {
    milvus::DatabaseDesc desc;
    auto status = client_->DescribeDatabase("test", desc);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, DescribeDatabaseFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string db_name = "test";

    EXPECT_CALL(service_, DescribeDatabase(_, Property(&DescribeDatabaseRequest::db_name, db_name), _))
        .WillOnce([](::grpc::ServerContext*, const DescribeDatabaseRequest*, DescribeDatabaseResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
        });

    milvus::DatabaseDesc desc;
    auto status = client_->DescribeDatabase(db_name, desc);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}