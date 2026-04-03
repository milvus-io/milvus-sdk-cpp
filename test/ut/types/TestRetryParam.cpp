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

#include "milvus/MilvusClientV2.h"

class RetryParamTest : public ::testing::Test {};

TEST_F(RetryParamTest, DefaultValues) {
    milvus::RetryParam param;
    EXPECT_EQ(param.MaxRetryTimes(), 75u);
    EXPECT_EQ(param.MaxRetryTimeoutMs(), 0u);
    EXPECT_EQ(param.InitialBackOffMs(), 10u);
    EXPECT_EQ(param.MaxBackOffMs(), 3000u);
    EXPECT_EQ(param.BackOffMultiplier(), 3u);
    EXPECT_TRUE(param.RetryOnRateLimit());
}

TEST_F(RetryParamTest, SetMaxRetryTimes) {
    milvus::RetryParam param;
    param.SetMaxRetryTimes(10);
    EXPECT_EQ(param.MaxRetryTimes(), 10u);
}

TEST_F(RetryParamTest, WithMaxRetryTimes) {
    milvus::RetryParam param;
    auto& ref = param.WithMaxRetryTimes(20);
    EXPECT_EQ(param.MaxRetryTimes(), 20u);
    EXPECT_EQ(&ref, &param);
}

TEST_F(RetryParamTest, SetMaxRetryTimeoutMs) {
    milvus::RetryParam param;
    param.SetMaxRetryTimeoutMs(5000);
    EXPECT_EQ(param.MaxRetryTimeoutMs(), 5000u);
}

TEST_F(RetryParamTest, WithMaxRetryTimeoutMs) {
    milvus::RetryParam param;
    auto& ref = param.WithMaxRetryTimeoutMs(6000);
    EXPECT_EQ(param.MaxRetryTimeoutMs(), 6000u);
    EXPECT_EQ(&ref, &param);
}

TEST_F(RetryParamTest, SetInitialBackOffMs) {
    milvus::RetryParam param;
    param.SetInitialBackOffMs(50);
    EXPECT_EQ(param.InitialBackOffMs(), 50u);
}

TEST_F(RetryParamTest, WithInitialBackOffMs) {
    milvus::RetryParam param;
    auto& ref = param.WithInitialBackOffMs(100);
    EXPECT_EQ(param.InitialBackOffMs(), 100u);
    EXPECT_EQ(&ref, &param);
}

TEST_F(RetryParamTest, SetMaxBackOffMs) {
    milvus::RetryParam param;
    param.SetMaxBackOffMs(5000);
    EXPECT_EQ(param.MaxBackOffMs(), 5000u);
}

TEST_F(RetryParamTest, WithMaxBackOffMs) {
    milvus::RetryParam param;
    auto& ref = param.WithMaxBackOffMs(8000);
    EXPECT_EQ(param.MaxBackOffMs(), 8000u);
    EXPECT_EQ(&ref, &param);
}

TEST_F(RetryParamTest, SetBackOffMultiplier) {
    milvus::RetryParam param;
    param.SetBackOffMultiplier(5);
    EXPECT_EQ(param.BackOffMultiplier(), 5u);
}

TEST_F(RetryParamTest, WithBackOffMultiplier) {
    milvus::RetryParam param;
    auto& ref = param.WithBackOffMultiplier(2);
    EXPECT_EQ(param.BackOffMultiplier(), 2u);
    EXPECT_EQ(&ref, &param);
}

TEST_F(RetryParamTest, SetRetryOnRateLimit) {
    milvus::RetryParam param;
    param.SetRetryOnRateLimit(false);
    EXPECT_FALSE(param.RetryOnRateLimit());
}

TEST_F(RetryParamTest, WithRetryOnRateLimit) {
    milvus::RetryParam param;
    auto& ref = param.WithRetryOnRateLimit(false);
    EXPECT_FALSE(param.RetryOnRateLimit());
    EXPECT_EQ(&ref, &param);
}

TEST_F(RetryParamTest, CopyAssignment) {
    milvus::RetryParam param1;
    param1.SetMaxRetryTimes(99);
    param1.SetRetryOnRateLimit(false);

    milvus::RetryParam param2;
    param2 = param1;
    EXPECT_EQ(param2.MaxRetryTimes(), 99u);
    EXPECT_FALSE(param2.RetryOnRateLimit());
}
