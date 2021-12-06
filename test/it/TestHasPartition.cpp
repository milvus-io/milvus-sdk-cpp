#include <gtest/gtest.h>

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::HasPartitionRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, HasPartitionFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    const std::string collection{"Foo"};
    const std::string partition{"Bar"};

    for (auto value : {true, false}) {
        EXPECT_CALL(service_, HasPartition(_,
                                           AllOf(Property(&HasPartitionRequest::collection_name, collection),

                                                 Property(&HasPartitionRequest::partition_name, partition)),
                                           _))
            .WillOnce([value](::grpc::ServerContext*, const HasPartitionRequest*,
                              ::milvus::proto::milvus::BoolResponse* response) {
                response->set_value(value);
                return ::grpc::Status{};
            });
        bool has{false};
        auto status = client_->HasPartition(collection, partition, has);

        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(has, value);
    }
}
