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

class FlushAllResponseTest : public ::testing::Test {};

TEST_F(FlushAllResponseTest, SetterAndGetter) {
    milvus::FlushAllResponse resp;
    EXPECT_EQ(resp.FlushAllTs(), 0);

    resp.SetFlushAllTs(12345);
    EXPECT_EQ(resp.FlushAllTs(), 12345);
}

class GetFlushAllStateResponseTest : public ::testing::Test {};

TEST_F(GetFlushAllStateResponseTest, SetterAndGetter) {
    milvus::GetFlushAllStateResponse resp;
    EXPECT_FALSE(resp.Flushed());

    resp.SetFlushed(true);
    EXPECT_TRUE(resp.Flushed());

    resp.SetFlushed(false);
    EXPECT_FALSE(resp.Flushed());
}

class CompactResponseTest : public ::testing::Test {};

TEST_F(CompactResponseTest, SetterAndGetter) {
    milvus::CompactResponse resp;
    resp.SetCompactionID(999);
    EXPECT_EQ(resp.CompactionID(), 999);

    resp.SetCompactionPlanCount(5);
    EXPECT_EQ(resp.CompactionPlanCount(), 5);
}

class OptimizeResponseTest : public ::testing::Test {};

TEST_F(OptimizeResponseTest, SetterAndGetter) {
    milvus::OptimizeResponse resp;

    resp.SetStatusText("success");
    EXPECT_EQ(resp.StatusText(), "success");
    resp.SetCollectionName("coll");
    EXPECT_EQ(resp.CollectionName(), "coll");
    resp.SetCompactionID(1001);
    EXPECT_EQ(resp.CompactionID(), 1001);
    resp.SetTargetSize("512MB");
    EXPECT_EQ(resp.TargetSize(), "512MB");

    resp.AddProgress("initializing");
    resp.AddProgress("compacting");
    ASSERT_EQ(resp.ProgressHistory().size(), 2);
    EXPECT_EQ(resp.ProgressHistory()[1], "compacting");

    std::vector<std::string> history{"done"};
    resp.SetProgressHistory(std::move(history));
    ASSERT_EQ(resp.ProgressHistory().size(), 1);
    EXPECT_EQ(resp.ProgressHistory()[0], "done");
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

    EXPECT_EQ(resp.CompactionID(), 0);
    resp.SetCompactionID(123);
    EXPECT_EQ(resp.CompactionID(), 123);

    EXPECT_EQ(resp.State(), milvus::CompactionStateCode::UNKNOWN);
    resp.SetState(milvus::CompactionStateCode::COMPLETED);
    EXPECT_EQ(resp.State(), milvus::CompactionStateCode::COMPLETED);

    EXPECT_TRUE(resp.CollectionName().empty());
    resp.SetCollectionName("coll");
    EXPECT_EQ(resp.CollectionName(), "coll");
}

class GetServerVersionResponseTest : public ::testing::Test {};

TEST_F(GetServerVersionResponseTest, SetterAndGetter) {
    milvus::GetServerVersionResponse resp;

    EXPECT_TRUE(resp.Version().empty());
    EXPECT_TRUE(resp.BuildTime().empty());
    EXPECT_TRUE(resp.GitCommit().empty());
    EXPECT_TRUE(resp.GoVersion().empty());
    EXPECT_TRUE(resp.DeployMode().empty());

    resp.SetVersion("v2.5.0");
    resp.SetBuildTime("2024-01-01");
    resp.SetGitCommit("abc123");
    resp.SetGoVersion("go1.21");
    resp.SetDeployMode("standalone");

    EXPECT_EQ(resp.Version(), "v2.5.0");
    EXPECT_EQ(resp.BuildTime(), "2024-01-01");
    EXPECT_EQ(resp.GitCommit(), "abc123");
    EXPECT_EQ(resp.GoVersion(), "go1.21");
    EXPECT_EQ(resp.DeployMode(), "standalone");
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

class ListRunningRequestsResponseTest : public ::testing::Test {};

TEST_F(ListRunningRequestsResponseTest, SetterAndGetter) {
    milvus::ListRunningRequestsResponse resp;

    EXPECT_TRUE(resp.Requests().empty());
    EXPECT_TRUE(resp.NodeResults().empty());

    milvus::RunningRequestInfos requests;
    milvus::RunningRequestInfo info;
    info.SetRequestId("7-to03zu-1");
    info.SetProxyId(7);
    info.SetType("Search");
    info.SetTaskIds(std::vector<int64_t>{11, 12});
    info.SetCancellable(true);
    requests.emplace_back(std::move(info));
    resp.SetRequests(std::move(requests));
    EXPECT_EQ(resp.Requests().size(), 1);
    EXPECT_EQ(resp.Requests()[0].RequestId(), "7-to03zu-1");
    EXPECT_EQ(resp.Requests()[0].ProxyId(), 7);
    EXPECT_EQ(resp.Requests()[0].Type(), "Search");
    EXPECT_EQ(resp.Requests()[0].TaskIds().size(), 2);
    EXPECT_TRUE(resp.Requests()[0].Cancellable());

    milvus::RunningRequestNodeResults node_results;
    milvus::RunningRequestNodeResult node_result;
    node_result.SetNodeId(7);
    node_result.SetErrorCode(0);
    node_results.emplace_back(std::move(node_result));
    resp.SetNodeResults(std::move(node_results));
    EXPECT_EQ(resp.NodeResults().size(), 1);
    EXPECT_EQ(resp.NodeResults()[0].NodeId(), 7);
    EXPECT_TRUE(resp.NodeResults()[0].IsSuccess());
}

class CancelRequestsResponseTest : public ::testing::Test {};

TEST_F(CancelRequestsResponseTest, SetterAndGetter) {
    milvus::CancelRequestsResponse resp;

    EXPECT_TRUE(resp.Canceled().empty());
    EXPECT_TRUE(resp.NotFound().empty());
    EXPECT_TRUE(resp.Undetermined().empty());
    EXPECT_TRUE(resp.NodeResults().empty());

    milvus::RunningRequestInfos canceled;
    milvus::RunningRequestInfo info;
    info.SetRequestId("7-to03zu-1");
    info.SetState("Running");
    canceled.emplace_back(std::move(info));
    resp.SetCanceled(std::move(canceled));
    EXPECT_EQ(resp.Canceled().size(), 1);
    EXPECT_EQ(resp.Canceled()[0].RequestId(), "7-to03zu-1");
    EXPECT_EQ(resp.Canceled()[0].State(), "Running");

    resp.SetNotFound(std::vector<std::string>{"7-to03zu-2"});
    EXPECT_EQ(resp.NotFound().size(), 1);
    EXPECT_EQ(resp.NotFound()[0], "7-to03zu-2");

    resp.SetUndetermined(std::vector<std::string>{"8-to03zv-3"});
    EXPECT_EQ(resp.Undetermined().size(), 1);
    EXPECT_EQ(resp.Undetermined()[0], "8-to03zv-3");

    milvus::RunningRequestNodeResults node_results;
    milvus::RunningRequestNodeResult node_result;
    node_result.SetNodeId(7);
    node_result.SetErrorCode(65535);
    node_result.SetReason("node unreachable");
    node_results.emplace_back(std::move(node_result));
    resp.SetNodeResults(std::move(node_results));
    EXPECT_EQ(resp.NodeResults().size(), 1);
    EXPECT_EQ(resp.NodeResults()[0].NodeId(), 7);
    EXPECT_FALSE(resp.NodeResults()[0].IsSuccess());
}

class RunningRequestInfoTest : public ::testing::Test {};

TEST_F(RunningRequestInfoTest, DefaultConstruction) {
    milvus::RunningRequestInfo info;

    EXPECT_TRUE(info.RequestId().empty());
    EXPECT_EQ(info.ProxyId(), 0);
    EXPECT_TRUE(info.Type().empty());
    EXPECT_TRUE(info.DatabaseName().empty());
    EXPECT_TRUE(info.CollectionName().empty());
    EXPECT_TRUE(info.User().empty());
    EXPECT_TRUE(info.ClientAddr().empty());
    EXPECT_EQ(info.Nq(), 0);
    EXPECT_EQ(info.TopK(), 0);
    EXPECT_TRUE(info.Expr().empty());
    EXPECT_EQ(info.StartTimeMs(), 0);
    EXPECT_EQ(info.QueuedMs(), 0);
    EXPECT_EQ(info.ElapsedMs(), 0);
    EXPECT_TRUE(info.State().empty());
    EXPECT_TRUE(info.TaskIds().empty());
    EXPECT_TRUE(info.TraceId().empty());
    EXPECT_FALSE(info.Cancellable());
}

TEST_F(RunningRequestInfoTest, AllFields) {
    milvus::RunningRequestInfo info;

    info.SetRequestId("25-to03zu-rs");
    EXPECT_EQ(info.RequestId(), "25-to03zu-rs");
    auto& req_ref = info.WithRequestId("26-to03zu-rs");
    EXPECT_EQ(&req_ref, &info);

    info.SetProxyId(7);
    EXPECT_EQ(info.ProxyId(), 7);
    info.WithProxyId(8);
    EXPECT_EQ(info.ProxyId(), 8);

    info.SetType("HybridSearch");
    EXPECT_EQ(info.Type(), "HybridSearch");

    info.SetDatabaseName("db");
    EXPECT_EQ(info.DatabaseName(), "db");
    info.SetCollectionName("coll");
    EXPECT_EQ(info.CollectionName(), "coll");
    info.SetUser("alice");
    EXPECT_EQ(info.User(), "alice");
    info.SetClientAddr("10.0.0.1:5000");
    EXPECT_EQ(info.ClientAddr(), "10.0.0.1:5000");

    info.SetNq(4000);
    EXPECT_EQ(info.Nq(), 4000);
    info.WithNq(5000);
    EXPECT_EQ(info.Nq(), 5000);

    info.SetTopK(100);
    EXPECT_EQ(info.TopK(), 100);

    info.SetExpr("id > 0");
    EXPECT_EQ(info.Expr(), "id > 0");

    info.SetStartTimeMs(1700000000000);
    EXPECT_EQ(info.StartTimeMs(), 1700000000000);
    info.SetQueuedMs(3);
    EXPECT_EQ(info.QueuedMs(), 3);
    info.SetElapsedMs(1500);
    EXPECT_EQ(info.ElapsedMs(), 1500);

    info.SetState("Running");
    EXPECT_EQ(info.State(), "Running");

    info.SetTaskIds(std::vector<int64_t>{11, 12});
    EXPECT_EQ(info.TaskIds().size(), 2);
    EXPECT_EQ(info.TaskIds()[0], 11);
    EXPECT_EQ(info.TaskIds()[1], 12);

    info.SetTraceId("trace-1");
    EXPECT_EQ(info.TraceId(), "trace-1");

    info.SetCancellable(true);
    EXPECT_TRUE(info.Cancellable());
    info.WithCancellable(false);
    EXPECT_FALSE(info.Cancellable());
}

class RunningRequestNodeResultTest : public ::testing::Test {};

TEST_F(RunningRequestNodeResultTest, DefaultConstruction) {
    milvus::RunningRequestNodeResult result;

    EXPECT_EQ(result.NodeId(), 0);
    EXPECT_EQ(result.ErrorCode(), 0);
    EXPECT_TRUE(result.Reason().empty());
    EXPECT_FALSE(result.Unimplemented());
    EXPECT_TRUE(result.IsSuccess());
}

TEST_F(RunningRequestNodeResultTest, AllFields) {
    milvus::RunningRequestNodeResult result;

    result.SetNodeId(7);
    EXPECT_EQ(result.NodeId(), 7);
    auto& node_ref = result.WithNodeId(8);
    EXPECT_EQ(&node_ref, &result);
    EXPECT_EQ(result.NodeId(), 8);

    result.SetErrorCode(65535);
    EXPECT_EQ(result.ErrorCode(), 65535);
    auto& code_ref = result.WithErrorCode(0);
    EXPECT_EQ(&code_ref, &result);
    EXPECT_EQ(result.ErrorCode(), 0);

    result.SetReason("node unreachable");
    EXPECT_EQ(result.Reason(), "node unreachable");
    auto& reason_ref = result.WithReason("down");
    EXPECT_EQ(&reason_ref, &result);
    EXPECT_EQ(result.Reason(), "down");

    result.SetUnimplemented(true);
    EXPECT_TRUE(result.Unimplemented());
    auto& unim_ref = result.WithUnimplemented(false);
    EXPECT_EQ(&unim_ref, &result);
    EXPECT_FALSE(result.Unimplemented());

    EXPECT_TRUE(result.IsSuccess());
    result.SetErrorCode(65535);
    EXPECT_FALSE(result.IsSuccess());
}
