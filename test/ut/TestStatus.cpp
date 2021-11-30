#include <gtest/gtest.h>
#include "Status.h"

class StatusTest: public ::testing::Test {

};

TEST_F(StatusTest, DefaultStatus) {
    milvus::Status status;
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(status.message(), "OK");
    EXPECT_EQ(status.code(), milvus::StatusCode::OK);
}

TEST_F(StatusTest, StatusOK) {
    milvus::Status status = milvus::Status::OK();
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(status.message(), "OK");
    EXPECT_EQ(status.code(), milvus::StatusCode::OK);
}

TEST_F(StatusTest, ConstructorWithFailed) {
    milvus::Status status{milvus::StatusCode::ServerFailed, "server failed"};
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.message(), "server failed");
    EXPECT_EQ(status.code(), milvus::StatusCode::ServerFailed);
}

TEST_F(StatusTest, CopyConstructor) {
    milvus::Status statusFoo{milvus::StatusCode::ServerFailed, "server failed"};
    auto status = milvus::Status(statusFoo);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.message(), "server failed");
    EXPECT_EQ(status.code(), milvus::StatusCode::ServerFailed);
}

TEST_F(StatusTest, MoveConstructor) {
    milvus::Status statusFoo{milvus::StatusCode::ServerFailed, "server failed"};
    auto status = milvus::Status(std::move(statusFoo));
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.message(), "server failed");
    EXPECT_EQ(status.code(), milvus::StatusCode::ServerFailed);

    EXPECT_EQ(statusFoo.message(), "");
}