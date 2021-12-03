#include <gtest/gtest.h>

#include "mocks/MilvusMockedTest.h"

TEST_F(MilvusMockedTest, ConnectSuccessful) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    auto status = client_->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());
}

// TODO(jibin):
//       If need using connectivity api to check if real connected.
//       Currently, Connect always return success due to not real connect does.
// ref: https://github.com/grpc/grpc/blob/master/doc/connectivity-semantics-and-api.md
TEST_F(MilvusMockedTest, ConnectFailed) {
    auto port = server_.ListenPort();
    milvus::ConnectParam connect_param{"127.0.0.1", ++port};
    auto status = client_->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());
}