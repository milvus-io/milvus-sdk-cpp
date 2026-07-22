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
#include <unordered_map>

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::AddCollectionFieldRequest;
using ::milvus::proto::milvus::AlterCollectionSchemaRequest;
using ::milvus::proto::milvus::AlterCollectionSchemaResponse;
using ::testing::_;

namespace {

std::shared_ptr<milvus::MilvusClientV2>
CreateConnectedV2Client(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ::milvus::proto::milvus::ConnectRequest*,
                     ::milvus::proto::milvus::ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{"127.0.0.1", port};
    auto status = client->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());
    return client;
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, AddCollectionFieldUsesAlterCollectionSchema) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field("new_field", milvus::DataType::INT64);
    field.SetNullable(true);

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest* request,
                     AlterCollectionSchemaResponse* response) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_TRUE(request->action().has_add_request());
            EXPECT_EQ(request->action().add_request().field_infos_size(), 1);
            EXPECT_EQ(request->action().add_request().func_schema_size(), 0);
            const auto& field_schema = request->action().add_request().field_infos(0).field_schema();
            EXPECT_EQ(field_schema.name(), "new_field");
            EXPECT_EQ(field_schema.data_type(), ::milvus::proto::schema::DataType::Int64);
            EXPECT_TRUE(field_schema.nullable());
            response->mutable_alter_status()->set_code(0);
            return ::grpc::Status{};
        });

    auto status = client->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithDatabaseName("db").WithCollectionName("coll").WithField(
            std::move(field)));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, AddCollectionFieldFallsBackOnUnimplemented) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field("new_field", milvus::DataType::INT64);
    field.SetNullable(true);
    ::testing::InSequence sequence;

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNIMPLEMENTED, "method not implemented"};
        });
    EXPECT_CALL(service_, AddCollectionField(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AddCollectionFieldRequest* request,
                     ::milvus::proto::common::Status* response) {
            EXPECT_EQ(request->db_name(), "db");
            EXPECT_EQ(request->collection_name(), "coll");
            ::milvus::proto::schema::FieldSchema field_schema;
            EXPECT_TRUE(field_schema.ParseFromString(request->schema()));
            EXPECT_EQ(field_schema.name(), "new_field");
            EXPECT_EQ(field_schema.data_type(), ::milvus::proto::schema::DataType::Int64);
            EXPECT_TRUE(field_schema.nullable());
            response->set_code(0);
            return ::grpc::Status{};
        });

    auto status = client->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithDatabaseName("db").WithCollectionName("coll").WithField(
            std::move(field)));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, AddCollectionFieldFallsBackOnServiceUnimplemented) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field("new_field", milvus::DataType::INT64);
    field.SetNullable(true);
    ::testing::InSequence sequence;

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse* response) {
                response->mutable_alter_status()->set_code(10);
                response->mutable_alter_status()->set_reason("service unimplemented");
                return ::grpc::Status{};
            });
    EXPECT_CALL(service_, AddCollectionField(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const AddCollectionFieldRequest*, ::milvus::proto::common::Status* response) {
                response->set_code(0);
                return ::grpc::Status{};
            });

    auto status = client->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithCollectionName("coll").WithField(std::move(field)));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, AddCollectionFieldDoesNotFallbackOnServerError) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field("new_field", milvus::DataType::INT64);
    field.SetNullable(true);

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse* response) {
                response->mutable_alter_status()->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
                response->mutable_alter_status()->set_reason("alter field failed");
                return ::grpc::Status{};
            });

    auto status = client->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithCollectionName("coll").WithField(std::move(field)));
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, AddCollectionFieldDoesNotFallbackOnServerCodeMatchingGrpcUnimplemented) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field("new_field", milvus::DataType::INT64);

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse* response) {
                response->mutable_alter_status()->set_code(static_cast<int32_t>(::grpc::StatusCode::UNIMPLEMENTED));
                response->mutable_alter_status()->set_reason("server rejected alter field");
                return ::grpc::Status{};
            });

    auto status = client->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithCollectionName("coll").WithField(std::move(field)));
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
    EXPECT_EQ(status.RpcErrCode(), 0);
    EXPECT_EQ(status.ServerCode(), static_cast<int32_t>(::grpc::StatusCode::UNIMPLEMENTED));
}

TEST_F(UnconnectMilvusMockedTest, AddCollectionFieldDoesNotFallbackOnOtherGrpcError) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field("new_field", milvus::DataType::INT64);
    field.SetNullable(true);

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse*) {
            return ::grpc::Status{::grpc::StatusCode::PERMISSION_DENIED, "permission denied"};
        });

    auto status = client->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithCollectionName("coll").WithField(std::move(field)));
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
    EXPECT_EQ(status.RpcErrCode(), static_cast<int32_t>(::grpc::StatusCode::PERMISSION_DENIED));
}

TEST_F(UnconnectMilvusMockedTest, AddCollectionFieldPropagatesLegacyServerError) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field("new_field", milvus::DataType::INT64);
    ::testing::InSequence sequence;

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNIMPLEMENTED, "method not implemented"};
        });
    EXPECT_CALL(service_, AddCollectionField(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const AddCollectionFieldRequest*, ::milvus::proto::common::Status* response) {
                response->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
                response->set_reason("legacy add field failed");
                return ::grpc::Status{};
            });

    auto status = client->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithCollectionName("coll").WithField(std::move(field)));
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
    EXPECT_EQ(status.Message(), "legacy add field failed");
}

TEST_F(UnconnectMilvusMockedTest, AddCollectionFieldPropagatesLegacyGrpcError) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field("new_field", milvus::DataType::INT64);
    ::testing::InSequence sequence;

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNIMPLEMENTED, "method not implemented"};
        });
    EXPECT_CALL(service_, AddCollectionField(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AddCollectionFieldRequest*, ::milvus::proto::common::Status*) {
            return ::grpc::Status{::grpc::StatusCode::PERMISSION_DENIED, "legacy permission denied"};
        });

    auto status = client->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithCollectionName("coll").WithField(std::move(field)));
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
    EXPECT_EQ(status.RpcErrCode(), static_cast<int32_t>(::grpc::StatusCode::PERMISSION_DENIED));
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldNotConnected) {
    auto client = milvus::MilvusClientV2::Create();
    auto status = client->DropCollectionField(
        milvus::DropCollectionFieldRequest().WithCollectionName("coll").WithFieldName("f1"));
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldServerFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse* response) {
                response->mutable_alter_status()->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
                response->mutable_alter_status()->set_reason("drop field failed");
                return ::grpc::Status{};
            });

    auto status = client->DropCollectionField(
        milvus::DropCollectionFieldRequest().WithCollectionName("coll").WithFieldName("f1"));
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldByName) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest* request,
                     AlterCollectionSchemaResponse* response) {
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_TRUE(request->action().has_drop_request());
            EXPECT_EQ(request->action().drop_request().field_name(), "f1");
            EXPECT_FALSE(request->action().drop_request().drop_function_output_fields());
            response->mutable_alter_status()->set_code(0);
            return ::grpc::Status{};
        });

    auto status = client->DropCollectionField(
        milvus::DropCollectionFieldRequest().WithCollectionName("coll").WithFieldName("f1"));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldById) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest* request,
                     AlterCollectionSchemaResponse* response) {
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_TRUE(request->action().has_drop_request());
            EXPECT_EQ(request->action().drop_request().field_id(), 101);
            response->mutable_alter_status()->set_code(0);
            return ::grpc::Status{};
        });

    auto status =
        client->DropCollectionField(milvus::DropCollectionFieldRequest().WithCollectionName("coll").WithFieldID(101));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldRejectsBothSelectors) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    auto status = client->DropCollectionField(
        milvus::DropCollectionFieldRequest().WithCollectionName("coll").WithFieldName("f1").WithFieldID(101));
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldRejectsMissingSelector) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    auto status = client->DropCollectionField(milvus::DropCollectionFieldRequest().WithCollectionName("coll"));
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, DropCollectionFieldRejectsNegativeId) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    auto status =
        client->DropCollectionField(milvus::DropCollectionFieldRequest().WithCollectionName("coll").WithFieldID(-1));
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionFieldNotConnected) {
    auto client = milvus::MilvusClientV2::Create();
    milvus::FieldSchema field;
    field.SetName("sparse_vec");
    field.SetDataType(milvus::DataType::SPARSE_FLOAT_VECTOR);
    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("sparse_vec");
    milvus::IndexDesc index("sparse_vec", "sparse_idx", milvus::IndexType::SPARSE_INVERTED_INDEX,
                            milvus::MetricType::BM25);

    auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                               .WithCollectionName("coll")
                                               .WithField(std::move(field))
                                               .WithFunction(function)
                                               .WithIndex(std::move(index)));
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionFieldServerFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field;
    field.SetName("sparse_vec");
    field.SetDataType(milvus::DataType::SPARSE_FLOAT_VECTOR);
    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("sparse_vec");
    milvus::IndexDesc index("sparse_vec", "sparse_idx", milvus::IndexType::SPARSE_INVERTED_INDEX,
                            milvus::MetricType::BM25);

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse* response) {
                response->mutable_alter_status()->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
                response->mutable_alter_status()->set_reason("add function field failed");
                return ::grpc::Status{};
            });

    auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                               .WithCollectionName("coll")
                                               .WithField(std::move(field))
                                               .WithFunction(function)
                                               .WithIndex(std::move(index)));
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionFieldRejectsMismatchedOutputName) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field;
    field.SetName("sparse_vec");
    field.SetDataType(milvus::DataType::SPARSE_FLOAT_VECTOR);
    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("other_vec");
    milvus::IndexDesc index("sparse_vec", "sparse_idx", milvus::IndexType::SPARSE_INVERTED_INDEX,
                            milvus::MetricType::BM25);

    auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                               .WithCollectionName("coll")
                                               .WithField(std::move(field))
                                               .WithFunction(function)
                                               .WithIndex(std::move(index)));
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionField) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field;
    field.SetName("sparse_vec");
    field.SetDataType(milvus::DataType::SPARSE_FLOAT_VECTOR);
    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("sparse_vec");
    milvus::IndexDesc index("sparse_vec", "sparse_idx", milvus::IndexType::SPARSE_INVERTED_INDEX,
                            milvus::MetricType::BM25);
    index.AddExtraParam("drop_ratio_build", "0.2");

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest* request,
                     AlterCollectionSchemaResponse* response) {
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_TRUE(request->action().has_add_request());
            EXPECT_EQ(request->action().add_request().field_infos_size(), 1);
            if (request->action().add_request().field_infos_size() != 1) {
                return ::grpc::Status{};
            }
            const auto& field_info = request->action().add_request().field_infos(0);
            EXPECT_EQ(field_info.field_schema().name(), "sparse_vec");
            EXPECT_TRUE(field_info.field_schema().is_function_output());
            EXPECT_EQ(field_info.index_name(), "sparse_idx");
            std::unordered_map<std::string, std::string> index_params;
            for (const auto& param : field_info.extra_params()) {
                index_params.emplace(param.key(), param.value());
            }
            EXPECT_EQ(index_params.at("index_type"), "SPARSE_INVERTED_INDEX");
            EXPECT_EQ(index_params.at("metric_type"), "BM25");
            EXPECT_EQ(index_params.at("drop_ratio_build"), "0.2");
            EXPECT_EQ(request->action().add_request().func_schema_size(), 1);
            if (request->action().add_request().func_schema_size() != 1) {
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->action().add_request().func_schema(0).name(), "bm25_fn");
            response->mutable_alter_status()->set_code(0);
            return ::grpc::Status{};
        });

    auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                               .WithCollectionName("coll")
                                               .WithField(std::move(field))
                                               .WithFunction(function)
                                               .WithIndex(std::move(index)));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionFieldWithMinHash) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field("minhash_vec", milvus::DataType::BINARY_VECTOR);
    field.WithDimension(512);
    auto function = std::make_shared<milvus::Function>("minhash_fn", milvus::FunctionType::MINHASH);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("minhash_vec");
    milvus::IndexDesc index("", "minhash_idx", milvus::IndexType::MINHASH_LSH, milvus::MetricType::MHJACCARD);
    index.AddExtraParam("mh_lsh_band", "16");

    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest* request,
                     AlterCollectionSchemaResponse* response) {
            const auto& add_request = request->action().add_request();
            EXPECT_EQ(add_request.field_infos_size(), 1);
            EXPECT_EQ(add_request.func_schema_size(), 1);
            if (add_request.field_infos_size() != 1 || add_request.func_schema_size() != 1) {
                return ::grpc::Status{};
            }
            const auto& field_info = add_request.field_infos(0);
            EXPECT_EQ(field_info.field_schema().data_type(), ::milvus::proto::schema::DataType::BinaryVector);
            EXPECT_EQ(field_info.index_name(), "minhash_idx");
            std::unordered_map<std::string, std::string> index_params;
            for (const auto& param : field_info.extra_params()) {
                index_params.emplace(param.key(), param.value());
            }
            EXPECT_EQ(index_params.at("index_type"), "MINHASH_LSH");
            EXPECT_EQ(index_params.at("metric_type"), "MHJACCARD");
            EXPECT_EQ(index_params.at("mh_lsh_band"), "16");
            EXPECT_EQ(add_request.func_schema(0).type(), ::milvus::proto::schema::FunctionType::MinHash);
            response->mutable_alter_status()->set_code(0);
            return ::grpc::Status{};
        });

    auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                               .WithCollectionName("coll")
                                               .WithField(std::move(field))
                                               .WithFunction(function)
                                               .WithIndex(std::move(index)));
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionFieldRejectsMissingBoundIndex) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field("sparse_vec", milvus::DataType::SPARSE_FLOAT_VECTOR);
    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("sparse_vec");

    auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                               .WithCollectionName("coll")
                                               .WithField(std::move(field))
                                               .WithFunction(function));
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionFieldRejectsAutoIndex) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field("sparse_vec", milvus::DataType::SPARSE_FLOAT_VECTOR);
    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("sparse_vec");
    milvus::IndexDesc index("sparse_vec", "sparse_idx", milvus::IndexType::AUTOINDEX, milvus::MetricType::BM25);

    auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                               .WithCollectionName("coll")
                                               .WithField(std::move(field))
                                               .WithFunction(function)
                                               .WithIndex(std::move(index)));
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionFieldRejectsMismatchedIndexField) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    milvus::FieldSchema field("sparse_vec", milvus::DataType::SPARSE_FLOAT_VECTOR);
    auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
    function->AddInputFieldName("text");
    function->AddOutputFieldName("sparse_vec");
    milvus::IndexDesc index("other_vec", "sparse_idx", milvus::IndexType::SPARSE_INVERTED_INDEX,
                            milvus::MetricType::BM25);

    auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                               .WithCollectionName("coll")
                                               .WithField(std::move(field))
                                               .WithFunction(function)
                                               .WithIndex(std::move(index)));
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
}

TEST_F(UnconnectMilvusMockedTest, AddFunctionFieldRejectsReservedIndexParam) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    for (const auto* reserved_key : {"index_type", "metric_type", "params"}) {
        SCOPED_TRACE(reserved_key);
        milvus::FieldSchema field("sparse_vec", milvus::DataType::SPARSE_FLOAT_VECTOR);
        auto function = std::make_shared<milvus::Function>("bm25_fn", milvus::FunctionType::BM25);
        function->AddInputFieldName("text");
        function->AddOutputFieldName("sparse_vec");
        milvus::IndexDesc index("sparse_vec", "sparse_idx", milvus::IndexType::SPARSE_INVERTED_INDEX,
                                milvus::MetricType::BM25);
        index.AddExtraParam(reserved_key, "reserved");

        auto status = client->AddFunctionField(milvus::AddFunctionFieldRequest()
                                                   .WithCollectionName("coll")
                                                   .WithField(std::move(field))
                                                   .WithFunction(function)
                                                   .WithIndex(std::move(index)));
        EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
    }
}

TEST_F(UnconnectMilvusMockedTest, DropFunctionFieldNotConnected) {
    auto client = milvus::MilvusClientV2::Create();
    auto status = client->DropFunctionField(
        milvus::DropFunctionFieldRequest().WithCollectionName("coll").WithFunctionName("bm25_fn"));
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(UnconnectMilvusMockedTest, DropFunctionFieldServerFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const AlterCollectionSchemaRequest*, AlterCollectionSchemaResponse* response) {
                response->mutable_alter_status()->set_code(::milvus::proto::common::ErrorCode::UnexpectedError);
                response->mutable_alter_status()->set_reason("drop function field failed");
                return ::grpc::Status{};
            });

    auto status = client->DropFunctionField(
        milvus::DropFunctionFieldRequest().WithCollectionName("coll").WithFunctionName("bm25_fn"));
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, DropFunctionField) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    EXPECT_CALL(service_, AlterCollectionSchema(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const AlterCollectionSchemaRequest* request,
                     AlterCollectionSchemaResponse* response) {
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_TRUE(request->action().has_drop_request());
            EXPECT_EQ(request->action().drop_request().function_name(), "bm25_fn");
            EXPECT_TRUE(request->action().drop_request().drop_function_output_fields());
            response->mutable_alter_status()->set_code(0);
            return ::grpc::Status{};
        });

    auto status = client->DropFunctionField(
        milvus::DropFunctionFieldRequest().WithCollectionName("coll").WithFunctionName("bm25_fn"));
    EXPECT_TRUE(status.IsOk());
}
