#include <gtest/gtest.h>

#include "Status.h"

class StatusTest : public ::testing::Test {};

TEST_F(StatusTest, DefaultStatus) {
    milvus::Status status;
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_EQ(status.Code(), milvus::StatusCode::OK);
}

TEST_F(StatusTest, StatusOK) {
    milvus::Status status = milvus::Status::OK();
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(status.Message(), "OK");
    EXPECT_EQ(status.Code(), milvus::StatusCode::OK);
}

TEST_F(StatusTest, ConstructorWithFailed) {
    milvus::Status status{milvus::StatusCode::SERVER_FAILED, "server failed"};
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Message(), "server failed");
    EXPECT_EQ(status.Code(), milvus::StatusCode::SERVER_FAILED);
}

TEST_F(StatusTest, CopyConstructor) {
    milvus::Status statusFoo{milvus::StatusCode::SERVER_FAILED, "server failed"};
    auto status = milvus::Status(statusFoo);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Message(), "server failed");
    EXPECT_EQ(status.Code(), milvus::StatusCode::SERVER_FAILED);
}

TEST_F(StatusTest, MoveConstructor) {
    milvus::Status statusFoo{milvus::StatusCode::SERVER_FAILED, "server failed"};
    auto status = milvus::Status(std::move(statusFoo));
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Message(), "server failed");
    EXPECT_EQ(status.Code(), milvus::StatusCode::SERVER_FAILED);
}