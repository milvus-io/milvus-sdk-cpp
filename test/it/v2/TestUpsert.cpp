// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <memory>

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"

using ::testing::_;

namespace {

std::shared_ptr<milvus::MilvusClientV2>
CreateConnectedV2Client(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ::milvus::proto::milvus::ConnectRequest*,
                     ::milvus::proto::milvus::ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    auto status = client->Connect(milvus::ConnectParam{"127.0.0.1", port});
    EXPECT_TRUE(status.IsOk());
    return client;
}

void
FillPartialUpdateSchema(milvus::proto::milvus::DescribeCollectionResponse* response, bool include_tags = true) {
    auto* schema = response->mutable_schema();

    auto* id = schema->add_fields();
    id->set_name("id");
    id->set_data_type(milvus::proto::schema::DataType::Int64);
    id->set_is_primary_key(true);

    if (include_tags) {
        auto* tags = schema->add_fields();
        tags->set_name("tags");
        tags->set_data_type(milvus::proto::schema::DataType::Array);
        tags->set_element_type(milvus::proto::schema::DataType::VarChar);
        auto* max_capacity = tags->add_type_params();
        max_capacity->set_key("max_capacity");
        max_capacity->set_value("16");
        auto* max_length = tags->add_type_params();
        max_length->set_key("max_length");
        max_length->set_value("64");
    }

    auto* vector = schema->add_fields();
    vector->set_name("vector");
    vector->set_data_type(milvus::proto::schema::DataType::FloatVector);
    auto* dim = vector->add_type_params();
    dim->set_key("dim");
    dim->set_value("2");
}

void
SetSchemaMismatch(milvus::proto::milvus::MutationResult* response) {
    response->mutable_status()->set_error_code(milvus::proto::common::ErrorCode::SchemaMismatch);
    response->mutable_status()->set_reason("schema mismatch");
}

}  // namespace

// V2 request-style Upsert coverage.
TEST_F(UnconnectMilvusMockedTest, UpsertFieldPartialUpdateOps) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillPartialUpdateSchema(response);
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, Upsert(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::UpsertRequest* request,
                     milvus::proto::milvus::MutationResult* response) {
            EXPECT_EQ(request->collection_name(), "partial_update_coll");
            EXPECT_TRUE(request->partial_update());
            if (request->field_ops_size() != 2) {
                ADD_FAILURE() << "Expected two field operations, got " << request->field_ops_size();
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->field_ops(0).field_name(), "id");
            EXPECT_EQ(request->field_ops(0).op(), milvus::proto::schema::FieldPartialUpdateOp::REPLACE);
            EXPECT_EQ(request->field_ops(1).field_name(), "tags");
            EXPECT_EQ(request->field_ops(1).op(), milvus::proto::schema::FieldPartialUpdateOp::ARRAY_APPEND);
            response->mutable_status()->set_code(0);
            response->set_upsert_cnt(1);
            return ::grpc::Status{};
        });

    milvus::EntityRow row;
    row["id"] = 1;
    row["tags"] = nlohmann::json::array({"new_tag"});

    milvus::UpsertRequest request;
    request.WithCollectionName("partial_update_coll")
        .AddRowData(std::move(row))
        .AddFieldOp(milvus::FieldPartialUpdateOp("id"))
        .AddFieldOp(milvus::FieldPartialUpdateOp("tags", milvus::FieldPartialUpdateOp::OpType::ARRAY_APPEND));

    milvus::UpsertResponse response;
    auto status = client->Upsert(request, response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
    EXPECT_EQ(response.Results().UpsertCount(), 1);
}

TEST_F(UnconnectMilvusMockedTest, UpsertColumnsWithImplicitPartialUpdate) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillPartialUpdateSchema(response);
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, Upsert(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::UpsertRequest* request,
                     milvus::proto::milvus::MutationResult* response) {
            EXPECT_TRUE(request->partial_update());
            if (request->fields_data_size() != 2) {
                ADD_FAILURE() << "Expected two field data columns, got " << request->fields_data_size();
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->fields_data(0).field_name(), "id");
            EXPECT_EQ(request->fields_data(1).field_name(), "tags");
            if (request->field_ops_size() != 1) {
                ADD_FAILURE() << "Expected one field operation, got " << request->field_ops_size();
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->field_ops(0).field_name(), "tags");
            EXPECT_EQ(request->field_ops(0).op(), milvus::proto::schema::FieldPartialUpdateOp::ARRAY_APPEND);
            response->mutable_status()->set_code(0);
            response->set_upsert_cnt(1);
            return ::grpc::Status{};
        });

    auto ids = std::make_shared<milvus::Int64FieldData>("id", std::vector<int64_t>{1});
    auto tags = std::make_shared<milvus::ArrayVarCharFieldData>(
        "tags", std::vector<milvus::ArrayVarCharFieldData::ElementT>{{"new_tag"}});

    milvus::UpsertRequest request;
    request.WithCollectionName("partial_update_coll")
        .WithColumnsData({ids, tags})
        .AddFieldOp(milvus::FieldPartialUpdateOp("tags", milvus::FieldPartialUpdateOp::OpType::ARRAY_APPEND));

    milvus::UpsertResponse response;
    auto status = client->Upsert(request, response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
    EXPECT_EQ(response.Results().UpsertCount(), 1);
}

TEST_F(UnconnectMilvusMockedTest, UpsertPropagatesServerErrorForOmittedFieldOpPayload) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillPartialUpdateSchema(response);
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, Upsert(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::UpsertRequest* request,
                     milvus::proto::milvus::MutationResult* response) {
            if (request->fields_data_size() != 1) {
                ADD_FAILURE() << "Expected one field data column, got " << request->fields_data_size();
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->fields_data(0).field_name(), "id");
            if (request->field_ops_size() != 1) {
                ADD_FAILURE() << "Expected one field operation, got " << request->field_ops_size();
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->field_ops(0).field_name(), "tags");
            EXPECT_EQ(request->field_ops(0).op(), milvus::proto::schema::FieldPartialUpdateOp::ARRAY_APPEND);
            response->mutable_status()->set_code(1100);
            response->mutable_status()->set_reason(
                "partial-update op targets field \"tags\" not present in fields_data: invalid parameter");
            return ::grpc::Status{};
        });

    auto ids = std::make_shared<milvus::Int64FieldData>("id", std::vector<int64_t>{1});
    milvus::UpsertRequest request;
    request.WithCollectionName("partial_update_coll")
        .WithColumnsData({ids})
        .AddFieldOp(milvus::FieldPartialUpdateOp("tags", milvus::FieldPartialUpdateOp::OpType::ARRAY_APPEND));

    milvus::UpsertResponse response;
    auto status = client->Upsert(request, response);
    EXPECT_EQ(status.Code(), milvus::StatusCode::SERVER_FAILED);
    EXPECT_EQ(status.ServerCode(), 1100);
    EXPECT_EQ(status.Message(),
              "partial-update op targets field \"tags\" not present in fields_data: invalid parameter");
}

TEST_F(UnconnectMilvusMockedTest, UpsertPropagatesServerErrorForUnknownFieldPartialUpdateOp) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillPartialUpdateSchema(response);
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, Upsert(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::UpsertRequest* request,
                     milvus::proto::milvus::MutationResult* response) {
            if (request->field_ops_size() != 1) {
                ADD_FAILURE() << "Expected one field operation, got " << request->field_ops_size();
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->field_ops(0).field_name(), "tags");
            EXPECT_EQ(static_cast<int>(request->field_ops(0).op()), 99);
            response->mutable_status()->set_code(1100);
            response->mutable_status()->set_reason("unsupported partial update op: 99: invalid parameter");
            return ::grpc::Status{};
        });

    milvus::UpsertRequest request;
    request.WithCollectionName("partial_update_coll")
        .AddFieldOp(milvus::FieldPartialUpdateOp("tags", static_cast<milvus::FieldPartialUpdateOp::OpType>(99)));

    milvus::UpsertResponse response;
    auto status = client->Upsert(request, response);
    EXPECT_EQ(status.Code(), milvus::StatusCode::SERVER_FAILED);
    EXPECT_EQ(status.ServerCode(), 1100);
    EXPECT_EQ(status.Message(), "unsupported partial update op: 99: invalid parameter");
}

TEST_F(UnconnectMilvusMockedTest, UpsertRejectsInconsistentColumnCounts) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillPartialUpdateSchema(response);
            return ::grpc::Status{};
        });

    auto ids = std::make_shared<milvus::Int64FieldData>("id", std::vector<int64_t>{1, 2});
    auto tags = std::make_shared<milvus::ArrayVarCharFieldData>(
        "tags", std::vector<milvus::ArrayVarCharFieldData::ElementT>{{"new_tag"}});

    milvus::UpsertRequest request;
    request.WithCollectionName("partial_update_coll").WithColumnsData({ids, tags});

    milvus::UpsertResponse response;
    auto status = client->Upsert(request, response);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "The row count of input fields is inconsistent");
}

TEST_F(UnconnectMilvusMockedTest, UpsertRowsRefreshStaleSchemaForNewField) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .Times(2)
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillPartialUpdateSchema(response, false);
            return ::grpc::Status{};
        })
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillPartialUpdateSchema(response);
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, Upsert(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::UpsertRequest* request,
                     milvus::proto::milvus::MutationResult* response) {
            EXPECT_TRUE(request->partial_update());
            if (request->fields_data_size() != 2) {
                ADD_FAILURE() << "Expected two field data columns, got " << request->fields_data_size();
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->fields_data(0).field_name(), "id");
            EXPECT_EQ(request->fields_data(1).field_name(), "tags");
            response->mutable_status()->set_code(0);
            response->set_upsert_cnt(1);
            return ::grpc::Status{};
        });

    milvus::EntityRow row;
    row["id"] = 1;
    row["tags"] = nlohmann::json::array({"new_tag"});

    milvus::UpsertRequest request;
    request.WithCollectionName("partial_update_coll").WithPartialUpdate(true).AddRowData(std::move(row));

    milvus::UpsertResponse response;
    auto status = client->Upsert(request, response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
    EXPECT_EQ(response.Results().UpsertCount(), 1);
}

TEST_F(UnconnectMilvusMockedTest, InsertRowsRefreshStaleSchemaForNewField) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .Times(2)
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillPartialUpdateSchema(response, false);
            return ::grpc::Status{};
        })
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillPartialUpdateSchema(response);
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, Insert(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::InsertRequest* request,
                     milvus::proto::milvus::MutationResult* response) {
            if (request->fields_data_size() != 3) {
                ADD_FAILURE() << "Expected three field data columns, got " << request->fields_data_size();
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->fields_data(0).field_name(), "id");
            EXPECT_EQ(request->fields_data(1).field_name(), "tags");
            EXPECT_EQ(request->fields_data(2).field_name(), "vector");
            response->mutable_status()->set_code(0);
            response->set_insert_cnt(1);
            return ::grpc::Status{};
        });

    milvus::EntityRow row;
    row["id"] = 1;
    row["tags"] = nlohmann::json::array({"new_tag"});
    row["vector"] = std::vector<float>{0.1f, 0.2f};

    milvus::InsertRequest request;
    request.WithCollectionName("partial_update_coll").AddRowData(std::move(row));

    milvus::InsertResponse response;
    auto status = client->Insert(request, response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
    EXPECT_EQ(response.Results().InsertCount(), 1);
}

TEST_F(UnconnectMilvusMockedTest, InsertRetriesSchemaMismatchOnlyOnce) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .Times(2)
        .WillRepeatedly([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                           milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillPartialUpdateSchema(response);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, Insert(_, _, _))
        .Times(2)
        .WillRepeatedly([](::grpc::ServerContext*, const milvus::proto::milvus::InsertRequest*,
                           milvus::proto::milvus::MutationResult* response) {
            SetSchemaMismatch(response);
            return ::grpc::Status{};
        });

    milvus::EntityRow row;
    row["id"] = 1;
    row["tags"] = nlohmann::json::array({"new_tag"});
    row["vector"] = std::vector<float>{0.1f, 0.2f};

    milvus::InsertResponse response;
    auto status = client->Insert(
        milvus::InsertRequest().WithCollectionName("partial_update_coll").AddRowData(std::move(row)), response);
    EXPECT_EQ(status.LegacyServerCode(), static_cast<int32_t>(milvus::proto::common::ErrorCode::SchemaMismatch));
}

TEST_F(UnconnectMilvusMockedTest, UpsertRetriesSchemaMismatchOnlyOnce) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .Times(2)
        .WillRepeatedly([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                           milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillPartialUpdateSchema(response);
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, Upsert(_, _, _))
        .Times(2)
        .WillRepeatedly([](::grpc::ServerContext*, const milvus::proto::milvus::UpsertRequest*,
                           milvus::proto::milvus::MutationResult* response) {
            SetSchemaMismatch(response);
            return ::grpc::Status{};
        });

    milvus::EntityRow row;
    row["id"] = 1;
    row["tags"] = nlohmann::json::array({"new_tag"});

    milvus::UpsertResponse response;
    auto status = client->Upsert(milvus::UpsertRequest()
                                     .WithCollectionName("partial_update_coll")
                                     .WithPartialUpdate(true)
                                     .AddRowData(std::move(row)),
                                 response);
    EXPECT_EQ(status.LegacyServerCode(), static_cast<int32_t>(milvus::proto::common::ErrorCode::SchemaMismatch));
}
