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
#include "utils/TypeUtils.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::DescribeIndexRequest;
using ::milvus::proto::milvus::DescribeIndexResponse;
using ::testing::_;
using ::testing::Property;
using ::testing::UnorderedElementsAreArray;

TEST_F(MilvusMockedTest, ListIndexes) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection_name = "test_collection";
    const std::vector<std::string> index_names = {"aa", "bb", "cc"};

    milvus::IndexDesc index_desc;
    EXPECT_CALL(service_, DescribeIndex(_, Property(&DescribeIndexRequest::collection_name, collection_name), _))
        .WillOnce([&index_names](::grpc::ServerContext*, const DescribeIndexRequest*, DescribeIndexResponse* response) {
            for (auto name : index_names) {
                auto* index_desc_ptr = response->add_index_descriptions();
                index_desc_ptr->set_index_name(name);
                index_desc_ptr->set_field_name(name);

                auto kv = index_desc_ptr->add_params();
                kv->set_key(milvus::INDEX_TYPE);
                kv->set_value(std::to_string(milvus::IndexType::IVF_FLAT));
                kv = index_desc_ptr->add_params();
                kv->set_key(milvus::METRIC_TYPE);
                kv->set_value(std::to_string(milvus::MetricType::L2));
            }
            return ::grpc::Status{};
        });

    std::vector<std::string> output_names;
    auto status = client_->ListIndexes(collection_name, "", output_names);
    EXPECT_TRUE(status.IsOk());
    EXPECT_THAT(output_names, UnorderedElementsAreArray(index_names));
}
