#include <gtest/gtest.h>

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::HasCollectionRequest;
using ::testing::_;
using ::testing::Property;

TEST_F(MilvusMockedTest, HasCollectionFoo) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_schema = "Foo";

    for (auto value : {true, false}) {
        EXPECT_CALL(service_, HasCollection(_, Property(&HasCollectionRequest::collection_name, collection_schema), _))
            .WillOnce([value](::grpc::ServerContext*, const HasCollectionRequest*,
                              ::milvus::proto::milvus::BoolResponse* response) {
                response->set_value(value);
                return ::grpc::Status{};
            });
        bool has_collection{false};
        auto status = client_->HasCollection(collection_schema, has_collection);

        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(has_collection, value);
    }
}
