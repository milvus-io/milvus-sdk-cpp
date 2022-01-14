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
using ::milvus::proto::milvus::CalcDistanceRequest;
using ::milvus::proto::milvus::CalcDistanceResults;
using ::testing::_;
using ::testing::ElementsAreArray;
using ::testing::Property;

TEST_F(MilvusMockedTest, CalcDistanceFloat) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    milvus::FloatVecFieldDataPtr vectors_1 = std::make_shared<milvus::FloatVecFieldData>();
    milvus::FloatVecFieldData::ElementT element_1 = {1.0, 2.0};
    milvus::FloatVecFieldData::ElementT element_2 = {3.0, 4.0};
    vectors_1->Add(element_1);
    vectors_1->Add(element_2);

    milvus::FloatVecFieldDataPtr vectors_2 = std::make_shared<milvus::FloatVecFieldData>();
    milvus::FloatVecFieldData::ElementT element_3 = {1.0, 2.0};
    milvus::FloatVecFieldData::ElementT element_4 = {3.0, 4.0};
    milvus::FloatVecFieldData::ElementT element_5 = {5.0, 6.0};
    vectors_2->Add(element_3);
    vectors_2->Add(element_4);
    vectors_2->Add(element_5);

    milvus::CalcDistanceArguments arguments;
    arguments.SetLeftVectors(milvus::FloatVecFieldDataPtr{vectors_1});
    arguments.SetRightVectors(milvus::FloatVecFieldDataPtr{vectors_2});
    arguments.SetMetricType("ip");

    EXPECT_CALL(service_, CalcDistance(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const CalcDistanceRequest* request, CalcDistanceResults* response) {
            // validate params
            auto& params = request->params();
            for (int i = 0; i < params.size(); ++i) {
                if (params[i].key() == "metric") {
                    EXPECT_EQ(params[i].value(), "IP");
                }
            }

            // validate float data
            auto& op_left = request->op_left();
            EXPECT_TRUE(op_left.has_data_array());
            auto& data_left = op_left.data_array();
            EXPECT_EQ(data_left.dim(), element_1.size());
            EXPECT_TRUE(data_left.has_float_vector());

            auto& op_right = request->op_right();
            EXPECT_TRUE(op_right.has_data_array());
            auto& data_right = op_right.data_array();
            EXPECT_EQ(data_right.dim(), element_1.size());
            EXPECT_TRUE(data_right.has_float_vector());

            auto compare_array = [](const ::milvus::proto::schema::VectorField& data_arr,
                                    milvus::FloatVecFieldData::ElementT& element, size_t offset) {
                auto& float_array = data_arr.float_vector();
                EXPECT_GE(float_array.data_size(), offset + element.size());
                for (auto i = 0; i < element.size(); ++i) {
                    EXPECT_FLOAT_EQ(float_array.data(offset + i), element[i]);
                }
            };

            compare_array(data_left, element_1, 0);
            compare_array(data_left, element_2, element_1.size());

            compare_array(data_right, element_3, 0);
            compare_array(data_right, element_4, element_3.size());
            compare_array(data_right, element_5, element_3.size() + element_4.size());

            // set response
            auto result = response->mutable_float_dist();
            for (int i = 0; i < (vectors_1->Count() * vectors_2->Count()); ++i) {
                result->add_data((float)i);
            }

            return ::grpc::Status{};
        });

    milvus::DistanceArray results;
    auto status = client_->CalcDistance(arguments, results);
    EXPECT_TRUE(status.IsOk());

    EXPECT_FALSE(results.IsIntegerDistance());
    auto& float_results = results.FloatDistanceArray();
    EXPECT_EQ(float_results.size(), vectors_1->Count());
    int count = 0;
    for (int i = 0; i < vectors_1->Count(); ++i) {
        EXPECT_EQ(float_results[i].size(), vectors_2->Count());
        for (int k = 0; k < vectors_2->Count(); ++k) {
            EXPECT_FLOAT_EQ(float_results[i][k], (float)count++);
        }
    }
}

TEST_F(MilvusMockedTest, CalcDistanceBinary) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    milvus::BinaryVecFieldDataPtr vectors_1 = std::make_shared<milvus::BinaryVecFieldData>();
    milvus::BinaryVecFieldData::ElementT element_1 = {1, 2};
    milvus::BinaryVecFieldData::ElementT element_2 = {3, 4};
    vectors_1->Add(element_1);
    vectors_1->Add(element_2);

    milvus::BinaryVecFieldDataPtr vectors_2 = std::make_shared<milvus::BinaryVecFieldData>();
    milvus::BinaryVecFieldData::ElementT element_3 = {5, 6};
    milvus::BinaryVecFieldData::ElementT element_4 = {7, 8};
    milvus::BinaryVecFieldData::ElementT element_5 = {9, 10};
    vectors_2->Add(element_3);
    vectors_2->Add(element_4);
    vectors_2->Add(element_5);

    milvus::CalcDistanceArguments arguments;
    arguments.SetLeftVectors(milvus::BinaryVecFieldDataPtr{vectors_1});
    arguments.SetRightVectors(milvus::BinaryVecFieldDataPtr{vectors_2});
    arguments.SetMetricType("Hamming");
    arguments.SetDimension(1);

    EXPECT_CALL(service_, CalcDistance(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const CalcDistanceRequest* request, CalcDistanceResults* response) {
            // validate metric
            auto& params = request->params();
            for (int i = 0; i < params.size(); ++i) {
                if (params[i].key() == "metric") {
                    EXPECT_EQ(params[i].value(), "HAMMING");
                }
            }

            // validate float data
            auto& op_left = request->op_left();
            EXPECT_TRUE(op_left.has_data_array());
            auto& data_left = op_left.data_array();
            EXPECT_EQ(data_left.dim(), 1);
            EXPECT_FALSE(data_left.has_float_vector());

            auto& op_right = request->op_right();
            EXPECT_TRUE(op_right.has_data_array());
            auto& data_right = op_right.data_array();
            EXPECT_EQ(data_right.dim(), 1);
            EXPECT_FALSE(data_right.has_float_vector());

            auto compare_array = [](const ::milvus::proto::schema::VectorField& data_arr,
                                    milvus::BinaryVecFieldData::ElementT& element, size_t offset) {
                auto& binary_array = data_arr.binary_vector();
                EXPECT_GE(binary_array.length(), offset + element.size());
                for (auto i = 0; i < element.size(); ++i) {
                    EXPECT_EQ(binary_array.data()[offset + i], element[i]);
                }
            };

            compare_array(data_left, element_1, 0);
            compare_array(data_left, element_2, element_1.size());

            compare_array(data_right, element_3, 0);
            compare_array(data_right, element_4, element_3.size());
            compare_array(data_right, element_5, element_3.size() + element_4.size());

            // set response
            auto result = response->mutable_int_dist();
            for (int i = 0; i < (vectors_1->Count() * vectors_2->Count()); ++i) {
                result->add_data(i);
            }

            return ::grpc::Status{};
        });

    milvus::DistanceArray results;
    auto status = client_->CalcDistance(arguments, results);
    EXPECT_TRUE(status.IsOk());

    EXPECT_TRUE(results.IsIntegerDistance());
    auto& int_results = results.IntDistanceArray();
    EXPECT_EQ(int_results.size(), vectors_1->Count());
    int count = 0;
    for (int i = 0; i < vectors_1->Count(); ++i) {
        EXPECT_EQ(int_results[i].size(), vectors_2->Count());
        for (int k = 0; k < vectors_2->Count(); ++k) {
            EXPECT_EQ(int_results[i][k], count++);
        }
    }
}

TEST_F(MilvusMockedTest, CalcDistanceIDArray) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::vector<int64_t> ids_1 = {1, 2};
    std::vector<int64_t> ids_2 = {3, 4, 5};

    std::string left_collection = "c1";
    std::string right_collection = "c2";
    std::vector<std::string> left_partitions = {"p1", "p2"};
    std::vector<std::string> right_partitions = {"p3"};

    milvus::CalcDistanceArguments arguments;
    arguments.SetLeftVectors(std::make_shared<milvus::Int64FieldData>("f1", std::vector<int64_t>{ids_1}),
                             left_collection, std::vector<std::string>{left_partitions});

    arguments.SetRightVectors(std::make_shared<milvus::Int64FieldData>("f2", std::vector<int64_t>{ids_2}),
                              right_collection, std::vector<std::string>{right_partitions});
    arguments.SetMetricType("l2");

    EXPECT_CALL(service_, CalcDistance(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const CalcDistanceRequest* request, CalcDistanceResults* response) {
            // validate metric
            auto& params = request->params();
            for (int i = 0; i < params.size(); ++i) {
                if (params[i].key() == "metric") {
                    EXPECT_EQ(params[i].value(), "L2");
                }
            }

            // validate id array
            auto& op_left = request->op_left();
            EXPECT_TRUE(op_left.has_id_array());
            auto& data_left = op_left.id_array();

            auto& op_right = request->op_right();
            EXPECT_TRUE(op_right.has_id_array());
            auto& data_right = op_right.id_array();

            auto compare_array = [&](const ::milvus::proto::milvus::VectorIDs& data_arr,
                                     const std::string& collection_name,
                                     const std::vector<std::string>& partition_names, const std::vector<int64_t>& ids) {
                EXPECT_EQ(data_arr.collection_name(), collection_name);
                EXPECT_EQ(data_arr.partition_names_size(), partition_names.size());
                for (int i = 0; i < partition_names.size(); ++i) {
                    EXPECT_EQ(data_arr.partition_names()[i], partition_names[i]);
                }

                EXPECT_TRUE(data_arr.has_id_array());
                auto& id_arr = data_arr.id_array();
                for (int i = 0; i < ids.size(); ++i) {
                    EXPECT_EQ(id_arr.int_id().data()[i], ids[i]);
                }
            };

            compare_array(data_left, left_collection, left_partitions, ids_1);
            compare_array(data_right, right_collection, right_partitions, ids_2);

            // set response
            auto result = response->mutable_float_dist();
            for (int i = 0; i < (ids_1.size() * ids_2.size()); ++i) {
                result->add_data((float)i);
            }

            return ::grpc::Status{};
        });

    milvus::DistanceArray results;
    auto status = client_->CalcDistance(arguments, results);
    EXPECT_TRUE(status.IsOk());

    EXPECT_FALSE(results.IsIntegerDistance());
    auto& float_results = results.FloatDistanceArray();
    EXPECT_EQ(float_results.size(), ids_1.size());
    int count = 0;
    for (int i = 0; i < ids_1.size(); ++i) {
        EXPECT_EQ(float_results[i].size(), ids_2.size());
        for (int k = 0; k < ids_2.size(); ++k) {
            EXPECT_FLOAT_EQ(float_results[i][k], (float)count++);
        }
    }
}

TEST_F(MilvusMockedTest, CalcDistanceFooConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    milvus::CalcDistanceArguments arguments;
    milvus::DistanceArray results;
    auto status = client_->CalcDistance(arguments, results);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}
