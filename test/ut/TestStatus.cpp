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

#include "milvus/Status.h"

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