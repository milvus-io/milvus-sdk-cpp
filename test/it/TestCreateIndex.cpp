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
using ::milvus::proto::milvus::CreateIndexRequest;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Property;

TEST_F(MilvusMockedTest, TestCreateIndex) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "test_collection";
    std::string field_name = "test_field";
    std::string index_name = "test_index";
    std::string db_name = "test_db";
    int64_t index_id = 0;
    std::unordered_map<std::string, std::string> params;
    params["nlist"] = "10";
    params["nprobe"] = "10";
    params["nbits"] = "10";
    params["nlevel"] = "10";

    milvus::IndexDesc index_desc(field_name, index_name, index_id, params);
    const auto progress_monitor = ::milvus::ProgressMonitor::NoWait();
    EXPECT_CALL(service_, CreateIndex(_,
                                      AllOf(Property(&CreateIndexRequest::collection_name, collection_name),
                                            Property(&CreateIndexRequest::field_name, field_name)),
                                      _))
        .WillOnce([](::grpc::ServerContext*, const CreateIndexRequest*, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });
    auto status = client_->CreateIndex(collection_name, index_desc, progress_monitor);
    EXPECT_TRUE(status.IsOk());
}