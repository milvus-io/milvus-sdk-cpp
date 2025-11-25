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

#include "milvus/types/CompactionState.h"

class CompactionStateTest : public ::testing::Test {};

TEST_F(CompactionStateTest, GeneralTesting) {
    int64_t executing = 1;
    int64_t timeout = 2;
    int64_t completed = 3;

    milvus::CompactionStateCode state_code = milvus::CompactionStateCode::EXECUTING;

    milvus::CompactionState state;
    state.SetState(state_code);
    state.SetExecutingPlan(executing);
    state.SetTimeoutPlan(timeout);
    state.SetCompletedPlan(completed);
    EXPECT_EQ(state_code, state.State());
    EXPECT_EQ(executing, state.ExecutingPlan());
    EXPECT_EQ(timeout, state.TimeoutPlan());
    EXPECT_EQ(completed, state.CompletedPlan());
}
