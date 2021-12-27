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

#include <memory>

#include "TypeUtils.h"
#include "mocks/MilvusMockedTest.h"

using ::milvus::FieldDataPtr;
using ::milvus::StatusCode;
using ::milvus::proto::milvus::QueryRequest;
using ::milvus::proto::milvus::QueryResults;
using ::milvus::proto::schema::BinaryVector;
using ::milvus::proto::schema::BoolArray;
using ::milvus::proto::schema::DataType;
using ::milvus::proto::schema::DoubleArray;
using ::milvus::proto::schema::FieldData;
using ::milvus::proto::schema::FloatVector;
using ::milvus::proto::schema::IntArray;
using ::milvus::proto::schema::LongArray;
using ::milvus::proto::schema::ScalarField;
using ::milvus::proto::schema::StringArray;
using ::milvus::proto::schema::VectorField;

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Property;

TEST_F(MilvusMockedTest, QueryFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    milvus::QueryArguments query_arguments{};
    milvus::QueryResults query_results{};

    query_arguments.SetCollectionName("foo");
    query_arguments.AddPartitionName("part1");
    query_arguments.AddPartitionName("part2");
    query_arguments.SetExpression("id > 100");
    query_arguments.SetGuaranteeTimestamp(10000);
    query_arguments.SetTravelTimestamp(20000);
    query_arguments.AddOutputField("age");
    query_arguments.AddOutputField("score");
    query_arguments.AddOutputField("signature");

    EXPECT_CALL(service_, Query(_,
                                AllOf(Property(&QueryRequest::collection_name, "foo"),
                                      Property(&QueryRequest::partition_names, ElementsAre("part1", "part2")),
                                      Property(&QueryRequest::expr, "id > 100"),
                                      Property(&QueryRequest::guarantee_timestamp, 10000),
                                      Property(&QueryRequest::travel_timestamp, 20000),
                                      Property(&QueryRequest::output_fields, ElementsAre("age", "score", "signature"))),
                                _))
        .WillOnce([](::grpc::ServerContext*, const QueryRequest* request, QueryResults* response) {
            // ret: age
            auto& age = *response->add_fields_data();
            age.set_field_name("age");
            age.set_type(DataType::Int8);
            auto& age_data = *(age.mutable_scalars()->mutable_int_data());
            age_data.add_data(20);
            age_data.add_data(21);

            // ret: score
            auto& score = *response->add_fields_data();
            score.set_field_name("score");
            score.set_type(DataType::Int64);
            auto& score_data = *(score.mutable_scalars()->mutable_long_data());
            score_data.add_data(200000);
            score_data.add_data(210000);

            // ret: signature
            auto& sig = *response->add_fields_data();
            sig.set_field_name("signature");
            sig.set_type(DataType::FloatVector);
            auto& vectors = (*sig.mutable_vectors());
            vectors.set_dim(3);
            auto& sig_data = *(sig.mutable_vectors()->mutable_float_vector());
            sig_data.add_data(0.1f);
            sig_data.add_data(0.2f);
            sig_data.add_data(0.3f);
            sig_data.add_data(0.4f);
            sig_data.add_data(0.5f);
            sig_data.add_data(0.6f);

            return ::grpc::Status{};
        });

    auto status = client_->Query(query_arguments, query_results);
    auto& fields = query_results.OutputFields();

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(fields.size(), 3);
    EXPECT_EQ(fields[0]->Name(), "age");
    EXPECT_EQ(fields[0]->Type(), milvus::DataType::INT8);
    EXPECT_EQ(fields[0]->Count(), 2);
    EXPECT_EQ(fields[1]->Name(), "score");
    EXPECT_EQ(fields[1]->Type(), milvus::DataType::INT64);
    EXPECT_EQ(fields[1]->Count(), 2);
    EXPECT_EQ(fields[2]->Name(), "signature");
    EXPECT_EQ(fields[2]->Type(), milvus::DataType::FLOAT_VECTOR);
    EXPECT_EQ(fields[2]->Count(), 2);

    auto field_age = std::dynamic_pointer_cast<milvus::Int8FieldData>(fields[0]);
    EXPECT_NE(field_age, nullptr);
    EXPECT_THAT(field_age->Data(), ElementsAre(20, 21));

    auto field_score = std::dynamic_pointer_cast<milvus::Int64FieldData>(fields[1]);
    EXPECT_NE(field_score, nullptr);
    EXPECT_THAT(field_score->Data(), ElementsAre(200000, 210000));

    auto field_sig = std::dynamic_pointer_cast<milvus::FloatVecFieldData>(fields[2]);
    EXPECT_NE(field_sig, nullptr);
    EXPECT_THAT(field_sig->Data(),
                ElementsAre(std::vector<float>{0.1f, 0.2f, 0.3f}, std::vector<float>{0.4f, 0.5f, 0.6f}));
}

TEST_F(MilvusMockedTest, QueryWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    milvus::QueryArguments query_arguments{};
    milvus::QueryResults query_results{};

    auto status = client_->Query(query_arguments, query_results);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}
