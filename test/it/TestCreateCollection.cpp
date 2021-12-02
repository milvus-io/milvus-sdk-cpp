#include <gtest/gtest.h>

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::CreateCollectionRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, CreateCollectionFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    milvus::CollectionSchema collection_schema("Foo");
    milvus::proto::milvus::CreateCollectionRequest rpc_request;
    rpc_request.set_collection_name("Foo");

    EXPECT_CALL(service_, CreateCollection(_, Property(&CreateCollectionRequest::collection_name, "Foo"), _))
        .WillOnce([](::grpc::ServerContext*, const CreateCollectionRequest* request, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });
    auto status = client_->CreateCollection(collection_schema);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, CreateCollectionFooWithoutConnect) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};

    milvus::CollectionSchema collection_schema("Foo");
    milvus::proto::milvus::CreateCollectionRequest rpc_request;
    rpc_request.set_collection_name("Foo");

    auto status = client_->CreateCollection(collection_schema);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
}

TEST_F(MilvusMockedTest, CreateCollectionFooFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    milvus::CollectionSchema collection_schema("Foo");
    milvus::proto::milvus::CreateCollectionRequest rpc_request;
    rpc_request.set_collection_name("Foo");

    auto error_code = milvus::proto::common::ErrorCode::UnexpectedError;
    EXPECT_CALL(service_, CreateCollection(_, Property(&CreateCollectionRequest::collection_name, "Foo"), _))
        .WillOnce([error_code](::grpc::ServerContext*, const CreateCollectionRequest* request,
                               ::milvus::proto::common::Status* status) {
            status->set_error_code(error_code);
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
        });
    auto status = client_->CreateCollection(collection_schema);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}
