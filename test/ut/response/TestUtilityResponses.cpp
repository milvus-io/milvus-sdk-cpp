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

class CheckHealthResponseTest : public ::testing::Test {};

TEST_F(CheckHealthResponseTest, SetterAndGetter) {
    milvus::CheckHealthResponse resp;

    resp.SetIsHealthy(true);
    EXPECT_TRUE(resp.IsHealthy());

    resp.SetIsHealthy(false);
    EXPECT_FALSE(resp.IsHealthy());

    std::vector<std::string> reasons{"node down", "timeout"};
    resp.SetReasons(std::move(reasons));
    EXPECT_EQ(resp.Reasons().size(), 2);
    EXPECT_EQ(resp.Reasons()[0], "node down");
}

class CompactResponseTest : public ::testing::Test {};

TEST_F(CompactResponseTest, SetterAndGetter) {
    milvus::CompactResponse resp;
    resp.SetCompactionID(999);
    EXPECT_EQ(resp.CompactionID(), 999);

    resp.SetCompactionPlanCount(5);
    EXPECT_EQ(resp.CompactionPlanCount(), 5);
}

class GetCompactionStateResponseTest : public ::testing::Test {};

TEST_F(GetCompactionStateResponseTest, SetterAndGetter) {
    milvus::GetCompactionStateResponse resp;
    milvus::CompactionState state;
    resp.SetState(state);
    (void)resp.State();
}

class GetCompactionPlansResponseTest : public ::testing::Test {};

TEST_F(GetCompactionPlansResponseTest, SetterAndGetter) {
    milvus::GetCompactionPlansResponse resp;
    milvus::CompactionPlans plans;
    resp.SetPlans(std::move(plans));
    (void)resp.Plans();
}

class ListPersistentSegmentsResponseTest : public ::testing::Test {};

TEST_F(ListPersistentSegmentsResponseTest, SetterAndGetter) {
    milvus::ListPersistentSegmentsResponse resp;
    milvus::SegmentsInfo info;
    resp.SetResult(std::move(info));
    (void)resp.Result();
}

class ListQuerySegmentsResponseTest : public ::testing::Test {};

TEST_F(ListQuerySegmentsResponseTest, SetterAndGetter) {
    milvus::ListQuerySegmentsResponse resp;
    milvus::QuerySegmentsInfo info;
    resp.SetResult(std::move(info));
    (void)resp.Result();
}

class RunAnalyzerResponseTest : public ::testing::Test {};

TEST_F(RunAnalyzerResponseTest, SetterAndGetter) {
    milvus::RunAnalyzerResponse resp;
    milvus::AnalyzerResults results;
    resp.SetResults(std::move(results));
    (void)resp.Results();
}

TEST_F(CheckHealthResponseTest, QuotaStates) {
    milvus::CheckHealthResponse resp;

    // Default is empty
    EXPECT_TRUE(resp.QuotaStates().empty());

    std::vector<std::string> states{"MemoryExhausted", "DiskQuotaExceeded"};
    resp.SetQuotaStates(std::move(states));
    EXPECT_EQ(resp.QuotaStates().size(), 2);
    EXPECT_EQ(resp.QuotaStates()[0], "MemoryExhausted");
    EXPECT_EQ(resp.QuotaStates()[1], "DiskQuotaExceeded");
}
