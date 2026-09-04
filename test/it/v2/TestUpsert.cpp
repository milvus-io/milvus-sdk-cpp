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
#include "../mocks/Utils.h"
#include "milvus/MilvusClientV2.h"
#include "utils/cache/CollectionTsCache.h"
#include "utils/cache/SchemaCache.h"

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

void
FillStringBackedSchema(milvus::proto::milvus::DescribeCollectionResponse* response) {
    auto* schema = response->mutable_schema();
    schema->set_name("string_backed_coll");
    response->set_collectionid(300);

    auto* id = schema->add_fields();
    id->set_name("id");
    id->set_data_type(milvus::proto::schema::DataType::Int64);
    id->set_is_primary_key(true);
    id->set_autoid(true);

    auto* body = schema->add_fields();
    body->set_name("body");
    body->set_data_type(milvus::proto::schema::DataType::Text);

    auto* geo = schema->add_fields();
    geo->set_name("geo");
    geo->set_data_type(milvus::proto::schema::DataType::Geometry);

    auto* tsz = schema->add_fields();
    tsz->set_name("tsz");
    tsz->set_data_type(milvus::proto::schema::DataType::Timestamptz);

    auto* arr_text = schema->add_fields();
    arr_text->set_name("arr_text");
    arr_text->set_data_type(milvus::proto::schema::DataType::Array);
    arr_text->set_element_type(milvus::proto::schema::DataType::Text);
    auto* max_capacity = arr_text->add_type_params();
    max_capacity->set_key("max_capacity");
    max_capacity->set_value("16");

    auto* vector = schema->add_fields();
    vector->set_name("vector");
    vector->set_data_type(milvus::proto::schema::DataType::FloatVector);
    auto* dim = vector->add_type_params();
    dim->set_key("dim");
    dim->set_value("2");
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

TEST_F(UnconnectMilvusMockedTest, UpsertPreservesEmptyRpcDatabaseAndNormalizesCacheKey) {
    const std::string collection_name = "serverless_upsert_coll";
    const std::string endpoint = "127.0.0.1:" + std::to_string(server_.ListenPort());
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([&collection_name](::grpc::ServerContext*,
                                     const milvus::proto::milvus::DescribeCollectionRequest* request,
                                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            EXPECT_EQ(request->db_name(), "");
            EXPECT_EQ(request->collection_name(), collection_name);
            FillPartialUpdateSchema(response);
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, Upsert(_, _, _))
        .WillOnce([&collection_name](::grpc::ServerContext*, const milvus::proto::milvus::UpsertRequest* request,
                                     milvus::proto::milvus::MutationResult* response) {
            EXPECT_EQ(request->db_name(), "");
            EXPECT_EQ(request->collection_name(), collection_name);
            response->mutable_status()->set_code(0);
            response->set_upsert_cnt(1);
            response->set_timestamp(100);
            return ::grpc::Status{};
        });

    milvus::EntityRow row;
    row["id"] = 1;
    row["tags"] = nlohmann::json::array({"new_tag"});

    milvus::UpsertResponse response;
    auto status = client->Upsert(
        milvus::UpsertRequest()
            .WithCollectionName(collection_name)
            .AddRowData(std::move(row))
            .AddFieldOp(milvus::FieldPartialUpdateOp("id"))
            .AddFieldOp(milvus::FieldPartialUpdateOp("tags", milvus::FieldPartialUpdateOp::OpType::ARRAY_APPEND)),
        response);
    ASSERT_TRUE(status.IsOk()) << status.Message();

    milvus::CollectionDescPtr empty_db_desc;
    milvus::CollectionDescPtr default_db_desc;
    EXPECT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint, "", collection_name, empty_db_desc));
    EXPECT_TRUE(milvus::SchemaCache::GetInstance().Get(endpoint, "default", collection_name, default_db_desc));
    EXPECT_EQ(empty_db_desc, default_db_desc);
    EXPECT_EQ(milvus::CollectionTsCache::GetInstance().Get(endpoint, "default", collection_name), 100);

    milvus::SchemaCache::GetInstance().Invalidate(endpoint, "", collection_name);
    milvus::CollectionTsCache::GetInstance().Invalidate(endpoint, "", collection_name);
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

TEST_F(UnconnectMilvusMockedTest, InsertStringBackedColumns) {
    // column-based insert of TEXT/GEOMETRY/TIMESTAMPTZ: the wire type is derived from the schema
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillStringBackedSchema(response);
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, Insert(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::InsertRequest* request,
                     milvus::proto::milvus::MutationResult* response) {
            EXPECT_EQ(request->collection_name(), "string_backed_coll");
            if (request->fields_data_size() != 5) {
                ADD_FAILURE() << "Expected 5 field data columns, got " << request->fields_data_size();
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->fields_data(0).field_name(), "body");
            EXPECT_EQ(request->fields_data(0).type(), milvus::proto::schema::DataType::Text);
            if (request->fields_data(0).scalars().string_data().data_size() != 2) {
                ADD_FAILURE() << "Expected 2 string data rows for body";
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->fields_data(0).scalars().string_data().data(0), "alpha");

            EXPECT_EQ(request->fields_data(1).field_name(), "geo");
            EXPECT_EQ(request->fields_data(1).type(), milvus::proto::schema::DataType::Geometry);
            if (request->fields_data(1).scalars().geometry_wkt_data().data_size() != 2) {
                ADD_FAILURE() << "Expected 2 geometry wkt rows for geo";
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->fields_data(1).scalars().geometry_wkt_data().data(0), "POINT (1 1)");

            EXPECT_EQ(request->fields_data(2).field_name(), "tsz");
            EXPECT_EQ(request->fields_data(2).type(), milvus::proto::schema::DataType::Timestamptz);
            if (request->fields_data(2).scalars().string_data().data_size() != 2) {
                ADD_FAILURE() << "Expected 2 string data rows for tsz";
                return ::grpc::Status{};
            }

            EXPECT_EQ(request->fields_data(3).field_name(), "arr_text");
            EXPECT_EQ(request->fields_data(3).scalars().array_data().element_type(),
                      milvus::proto::schema::DataType::Text);
            if (request->fields_data(3).scalars().array_data().data_size() != 2) {
                ADD_FAILURE() << "Expected 2 array data rows for arr_text";
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->fields_data(3).scalars().array_data().data(0).string_data().data(0), "a");

            EXPECT_EQ(request->fields_data(4).field_name(), "vector");
            response->mutable_status()->set_code(0);
            response->set_insert_cnt(2);
            return ::grpc::Status{};
        });

    auto body = std::make_shared<milvus::VarCharFieldData>("body", std::vector<std::string>{"alpha", "beta"});
    auto geo =
        std::make_shared<milvus::VarCharFieldData>("geo", std::vector<std::string>{"POINT (1 1)", "POINT (2 2)"});
    auto tsz = std::make_shared<milvus::VarCharFieldData>(
        "tsz", std::vector<std::string>{"2025-01-01T00:00:00+00:00", "2025-01-02T00:00:00+00:00"});
    auto arr_text = std::make_shared<milvus::ArrayVarCharFieldData>(
        "arr_text", std::vector<std::vector<std::string>>{{"a", "b"}, {"c"}});
    auto vector = std::make_shared<milvus::FloatVecFieldData>(
        "vector", std::vector<std::vector<float>>{{0.1f, 0.2f}, {0.3f, 0.4f}});

    milvus::InsertResponse response;
    auto status = client->Insert(milvus::InsertRequest()
                                     .WithCollectionName("string_backed_coll")
                                     .WithColumnsData({body, geo, tsz, arr_text, vector}),
                                 response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
    EXPECT_EQ(response.Results().InsertCount(), 2);
}

TEST_F(UnconnectMilvusMockedTest, UpsertStringBackedColumns) {
    // column-based upsert of a TEXT field: the wire type is derived from the schema
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                     milvus::proto::milvus::DescribeCollectionResponse* response) {
            FillStringBackedSchema(response);
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, Upsert(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::UpsertRequest* request,
                     milvus::proto::milvus::MutationResult* response) {
            EXPECT_EQ(request->collection_name(), "string_backed_coll");
            if (request->fields_data_size() != 6) {
                ADD_FAILURE() << "Expected 6 field data columns, got " << request->fields_data_size();
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->fields_data(1).field_name(), "body");
            EXPECT_EQ(request->fields_data(1).type(), milvus::proto::schema::DataType::Text);
            if (request->fields_data(1).scalars().string_data().data_size() != 1) {
                ADD_FAILURE() << "Expected 1 string data row for body";
                return ::grpc::Status{};
            }
            EXPECT_EQ(request->fields_data(1).scalars().string_data().data(0), "alpha");

            response->mutable_status()->set_code(0);
            response->set_upsert_cnt(1);
            return ::grpc::Status{};
        });

    auto id = std::make_shared<milvus::Int64FieldData>("id", std::vector<int64_t>{1});
    auto body = std::make_shared<milvus::VarCharFieldData>("body", std::vector<std::string>{"alpha"});
    auto geo = std::make_shared<milvus::VarCharFieldData>("geo", std::vector<std::string>{"POINT (1 1)"});
    auto tsz = std::make_shared<milvus::VarCharFieldData>("tsz", std::vector<std::string>{"2025-01-01T00:00:00+00:00"});
    auto arr_text =
        std::make_shared<milvus::ArrayVarCharFieldData>("arr_text", std::vector<std::vector<std::string>>{{"a"}});
    auto vector = std::make_shared<milvus::FloatVecFieldData>("vector", std::vector<std::vector<float>>{{0.1f, 0.2f}});

    milvus::UpsertResponse response;
    auto status = client->Upsert(milvus::UpsertRequest()
                                     .WithCollectionName("string_backed_coll")
                                     .WithColumnsData({id, body, geo, tsz, arr_text, vector}),
                                 response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
    EXPECT_EQ(response.Results().UpsertCount(), 1);
}

TEST_F(UnconnectMilvusMockedTest, BuildFieldsDataStringBacked) {
    // exercises the TEXT/array-of-TEXT (and geometry/timestamptz) branches of BuildFieldsData
    milvus::CollectionSchema schema("string_backed_coll");
    schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "", true, true));
    schema.AddField(milvus::FieldSchema("body", milvus::DataType::TEXT));
    schema.AddField(milvus::FieldSchema("geo", milvus::DataType::GEOMETRY));
    schema.AddField(milvus::FieldSchema("tsz", milvus::DataType::TIMESTAMPTZ));
    schema.AddField(milvus::FieldSchema("arr_text", milvus::DataType::ARRAY)
                        .WithElementType(milvus::DataType::TEXT)
                        .WithMaxCapacity(16));
    schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(2));

    std::vector<milvus::FieldDataPtr> fields_data;
    milvus::BuildFieldsData(schema, fields_data, 4);
    ASSERT_EQ(fields_data.size(), 6u);

    auto body = std::dynamic_pointer_cast<milvus::VarCharFieldData>(fields_data[1]);
    ASSERT_NE(body, nullptr);
    ASSERT_EQ(body->Data().size(), 4u);
    EXPECT_EQ(body->Value(0), "text_0");

    auto geo = std::dynamic_pointer_cast<milvus::VarCharFieldData>(fields_data[2]);
    ASSERT_NE(geo, nullptr);
    EXPECT_EQ(geo->Data().size(), 4u);

    auto tsz = std::dynamic_pointer_cast<milvus::VarCharFieldData>(fields_data[3]);
    ASSERT_NE(tsz, nullptr);
    EXPECT_EQ(tsz->Data().size(), 4u);

    auto arr_text = std::dynamic_pointer_cast<milvus::ArrayVarCharFieldData>(fields_data[4]);
    ASSERT_NE(arr_text, nullptr);
    ASSERT_EQ(arr_text->Data().size(), 4u);
    EXPECT_EQ(arr_text->Value(0).size(), 2u);
}

TEST_F(UnconnectMilvusMockedTest, InsertEmptyDataShortCircuitsNoRpc) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::InsertResponse response;
    auto status = client->Insert(milvus::InsertRequest().WithCollectionName("empty_coll"), response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
    EXPECT_EQ(response.Results().InsertCount(), 0);
    EXPECT_EQ(response.Results().IdArray().GetRowCount(), 0);
}

TEST_F(UnconnectMilvusMockedTest, UpsertEmptyDataShortCircuitsNoRpc) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::UpsertResponse response;
    auto status = client->Upsert(milvus::UpsertRequest().WithCollectionName("empty_coll"), response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
    EXPECT_EQ(response.Results().UpsertCount(), 0);
    EXPECT_EQ(response.Results().IdArray().GetRowCount(), 0);
}

TEST_F(UnconnectMilvusMockedTest, GetEmptyIdsShortCircuitsNoRpc) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    milvus::GetResponse response;
    auto status = client->Get(milvus::GetRequest().WithCollectionName("empty_coll"), response);
    EXPECT_TRUE(status.IsOk()) << status.Message();
    EXPECT_EQ(response.Results().GetRowCount(), 0);
}
