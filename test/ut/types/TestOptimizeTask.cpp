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

#include "milvus/types/OptimizeTask.h"

class OptimizeTaskTest : public ::testing::Test {};

TEST_F(OptimizeTaskTest, TimeoutBeforeCompletion) {
    auto task = std::make_shared<milvus::OptimizeTask>();
    milvus::OptimizeResponse response;

    auto status = task->GetResult(response, 1);
    EXPECT_EQ(status.Code(), milvus::StatusCode::TIMEOUT);
    EXPECT_FALSE(task->IsDone());
}

TEST_F(OptimizeTaskTest, CancelRequestsCancellation) {
    auto task = std::make_shared<milvus::OptimizeTask>();

    EXPECT_TRUE(task->Cancel());
    EXPECT_FALSE(task->IsDone());
    EXPECT_TRUE(task->IsCancelled());
    EXPECT_EQ(task->CurrentProgress(), "cancelling");

    milvus::OptimizeResponse response;
    auto status = task->GetResult(response, 1);
    EXPECT_EQ(status.Code(), milvus::StatusCode::TIMEOUT);
    EXPECT_FALSE(task->IsDone());
}
