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
using ::milvus::proto::milvus::InsertRequest;
using ::milvus::proto::milvus::MutationResult;
using ::milvus::proto::schema::BinaryVector;
using ::milvus::proto::schema::BoolArray;
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
using ::testing::ElementsAreArray;
using ::testing::Property;

namespace {

const std::string collection{"Foo"};
const std::string partition{"Part"};

constexpr size_t num_of_rows{4};

auto bool_field_ptr = std::make_shared<::milvus::BoolFieldData>("bool", std::vector<bool>{true, true, false, false});
auto int8_field_ptr = std::make_shared<::milvus::Int8FieldData>("int8", std::vector<int8_t>{0, 1, 2, 3});
auto int16_field_ptr = std::make_shared<::milvus::Int16FieldData>("int16", std::vector<int16_t>{1, 2, 3, 4});
auto int32_field_ptr = std::make_shared<::milvus::Int32FieldData>("int32", std::vector<int32_t>{2, 3, 4, 5});
auto int64_field_ptr = std::make_shared<::milvus::Int64FieldData>("int64", std::vector<int64_t>{3, 4, 5, 6});
auto float_field_ptr = std::make_shared<::milvus::FloatFieldData>("float", std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f});
auto double_field_ptr =
    std::make_shared<::milvus::DoubleFieldData>("double", std::vector<double>{0.11, 0.22, 0.33, 0.44});
auto string_field_ptr =
    std::make_shared<::milvus::StringFieldData>("string", std::vector<std::string>{"a", "b", "c", "d"});
auto bins_field_ptr = std::make_shared<::milvus::BinaryVecFieldData>(
    "bins",
    std::vector<std::vector<uint8_t>>{
        {0x00, 0x11, 0xee, 0xff}, {0x00, 0x11, 0xee, 0xff}, {0x00, 0x11, 0xee, 0xff}, {0x00, 0x11, 0xee, 0xff}});
auto floats_field_ptr = std::make_shared<::milvus::FloatVecFieldData>(
    "floats",
    std::vector<std::vector<float>>{
        {0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}});

std::vector<FieldDataPtr> fields{bool_field_ptr,  int8_field_ptr,  int16_field_ptr,  int32_field_ptr,
                                 int64_field_ptr, float_field_ptr, double_field_ptr, string_field_ptr,
                                 bins_field_ptr,  floats_field_ptr};
}  // namespace

TEST_F(MilvusMockedTest, InsertFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::vector<int64_t> ret_ids{1000, 10001, 1002, 1003};

    EXPECT_CALL(service_,
                Insert(_,
                       AllOf(Property(&InsertRequest::collection_name, collection),
                             Property(&InsertRequest::partition_name, partition),
                             Property(&InsertRequest::num_rows, num_of_rows),
                             Property(&InsertRequest::fields_data_size, fields.size()),
                             Property(&InsertRequest::fields_data,
                                      ElementsAre(*bool_field_ptr, *int8_field_ptr, *int16_field_ptr, *int32_field_ptr,
                                                  *int64_field_ptr, *float_field_ptr, *double_field_ptr,
                                                  *string_field_ptr, *bins_field_ptr, *floats_field_ptr))),
                       _))
        .WillOnce([&ret_ids](::grpc::ServerContext*, const InsertRequest* request, MutationResult* response) {
            // ret
            for (const auto ret_id : ret_ids) {
                response->mutable_ids()->mutable_int_id()->add_data(ret_id);
            }
            return ::grpc::Status{};
        });

    milvus::DmlResults results;
    auto status = client_->Insert(collection, partition, fields, results);

    EXPECT_TRUE(status.IsOk());
    EXPECT_TRUE(results.IdArray().IsIntegerID());
    EXPECT_THAT(results.IdArray().IntIDArray(), ElementsAreArray(ret_ids));
}

TEST_F(MilvusMockedTest, InsertWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    milvus::DmlResults results;
    auto status = client_->Insert(collection, partition, fields, results);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}
