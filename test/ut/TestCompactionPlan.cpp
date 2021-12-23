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

#include "types/CompactionPlan.h"

class CompactionPlanTest : public ::testing::Test {};

TEST_F(CompactionPlanTest, GeneralTesting) {
    std::vector<int64_t> src_segments = {1, 2, 3};
    int64_t dst_segment = 4;

    milvus::CompactionPlan plan(src_segments, dst_segment);
    EXPECT_EQ(3, plan.SourceSegments().size());
    EXPECT_EQ(dst_segment, plan.DestinySegemnt());
}
