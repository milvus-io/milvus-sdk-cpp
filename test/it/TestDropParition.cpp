#include <gtest/gtest.h>

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::DropPartitionRequest;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Property;

TEST_F(MilvusMockedTest, DropPartitionFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    const std::string partition{"Bar"};

    EXPECT_CALL(service_, DropPartition(_,
                                        AllOf(Property(&DropPartitionRequest::collection_name, collection),

                                              Property(&DropPartitionRequest::partition_name, partition)),
                                        _))
        .WillOnce([](::grpc::ServerContext*, const DropPartitionRequest*, ::milvus::proto::common::Status*) {
            return ::grpc::Status{};
        });
    auto status = client_->DropPartition(collection, partition);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(MilvusMockedTest, DropPartitionFooFailed) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    const std::string partition{"Bar"};

    auto error_code = milvus::proto::common::ErrorCode::UnexpectedError;
    EXPECT_CALL(service_, DropPartition(_,
                                        AllOf(Property(&DropPartitionRequest::collection_name, collection),

                                              Property(&DropPartitionRequest::partition_name, partition)),
                                        _))
        .WillOnce([error_code](::grpc::ServerContext*, const DropPartitionRequest* request,
                               ::milvus::proto::common::Status* status) {
            status->set_error_code(error_code);
            return ::grpc::Status{::grpc::StatusCode::UNKNOWN, ""};
        });
    auto status = client_->DropPartition(collection, partition);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::SERVER_FAILED);
}
