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

#include <stdexcept>

#include "utils/RpcUtils.h"

class RpcUtilsTest : public ::testing::Test {};

TEST_F(RpcUtilsTest, StatusFromException) {
    auto status = milvus::StatusFromException(std::runtime_error("boom"));
    EXPECT_EQ(status.Code(), milvus::StatusCode::UNKNOWN_ERROR);
    EXPECT_EQ(status.Message(), "Unexpected SDK exception: boom");

    auto prefixed = milvus::StatusFromException(std::runtime_error("boom"), "Schema loader failed: ");
    EXPECT_EQ(prefixed.Code(), milvus::StatusCode::UNKNOWN_ERROR);
    EXPECT_EQ(prefixed.Message(), "Schema loader failed: boom");
}

TEST_F(RpcUtilsTest, StatusFromUnknownException) {
    auto status = milvus::StatusFromUnknownException();
    EXPECT_EQ(status.Code(), milvus::StatusCode::UNKNOWN_ERROR);
    EXPECT_EQ(status.Message(), "Unexpected SDK exception");

    auto custom = milvus::StatusFromUnknownException("Schema loader failed with unknown exception");
    EXPECT_EQ(custom.Code(), milvus::StatusCode::UNKNOWN_ERROR);
    EXPECT_EQ(custom.Message(), "Schema loader failed with unknown exception");
}
