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

#include "milvus/types/RestoreSnapshotJobInfo.h"
#include "milvus/types/RestoreSnapshotState.h"

class RestoreSnapshotJobInfoTest : public ::testing::Test {};

TEST_F(RestoreSnapshotJobInfoTest, SetterAndGetter) {
    milvus::RestoreSnapshotJobInfo info;
    info.SetJobID(101);
    info.SetSnapshotName("snap");
    info.SetDatabaseName("db");
    info.SetCollectionName("coll");
    info.SetState(milvus::RestoreSnapshotStateCode::EXECUTING);
    info.SetProgress(55);
    info.SetReason("running");
    info.SetStartTime(1000);
    info.SetTimeCost(2000);

    EXPECT_EQ(info.JobID(), 101);
    EXPECT_EQ(info.SnapshotName(), "snap");
    EXPECT_EQ(info.DatabaseName(), "db");
    EXPECT_EQ(info.CollectionName(), "coll");
    EXPECT_EQ(info.State(), milvus::RestoreSnapshotStateCode::EXECUTING);
    EXPECT_EQ(info.Progress(), 55);
    EXPECT_EQ(info.Reason(), "running");
    EXPECT_EQ(info.StartTime(), 1000);
    EXPECT_EQ(info.TimeCost(), 2000);
}

class RestoreSnapshotStateCodeTest : public ::testing::Test {};

TEST_F(RestoreSnapshotStateCodeTest, ToString) {
    EXPECT_EQ(std::to_string(milvus::RestoreSnapshotStateCode::UNKNOWN), "UNKNOWN");
    EXPECT_EQ(std::to_string(milvus::RestoreSnapshotStateCode::PENDING), "PENDING");
    EXPECT_EQ(std::to_string(milvus::RestoreSnapshotStateCode::EXECUTING), "EXECUTING");
    EXPECT_EQ(std::to_string(milvus::RestoreSnapshotStateCode::COMPLETED), "COMPLETED");
    EXPECT_EQ(std::to_string(milvus::RestoreSnapshotStateCode::FAILED), "FAILED");
}
