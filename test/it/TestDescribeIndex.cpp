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

#include "TypeUtils.h"
#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::DescribeIndexRequest;
using ::milvus::proto::milvus::DescribeIndexResponse;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, DescribeIndexFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection_name = "test_collection";
    const std::string field_name = "test_field";
    const std::string index_name = "test_index";

    DescribeIndexRequest request;
    request.set_collection_name(collection_name);
    request.set_field_name(field_name);

    milvus::IndexDesc index_desc;
    EXPECT_CALL(service_, DescribeIndex(_,
                                        AnyOf(Property(&DescribeIndexRequest::collection_name, collection_name),
                                              Property(&DescribeIndexRequest::field_name, field_name)),
                                        _))
        .WillOnce([&field_name, &index_name](::grpc::ServerContext*, const DescribeIndexRequest*,
                                             DescribeIndexResponse* response) {
            auto* index_desc_ptr = response->add_index_descriptions();
            index_desc_ptr->set_index_name(index_name);
            index_desc_ptr->set_field_name(field_name);

            auto kv = index_desc_ptr->add_params();
            kv->set_key(milvus::KeyIndexType());
            kv->set_value(std::to_string(milvus::IndexType::IVF_FLAT));
            kv = index_desc_ptr->add_params();
            kv->set_key(milvus::KeyMetricType());
            kv->set_value(std::to_string(milvus::MetricType::L2));

            kv = index_desc_ptr->add_params();
            kv->set_key(milvus::KeyParams());
            kv->set_value("{\"nlist\": 1024}");

            return ::grpc::Status{};
        });

    auto status = client_->DescribeIndex(collection_name, field_name, index_desc);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(index_desc.FieldName(), field_name);
    EXPECT_EQ(index_desc.IndexName(), index_name);
    EXPECT_EQ(index_desc.IndexType(), milvus::IndexType::IVF_FLAT);
    EXPECT_EQ(index_desc.MetricType(), milvus::MetricType::L2);
}