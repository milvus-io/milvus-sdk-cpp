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

class RefreshExternalCollectionJobInfoTest : public ::testing::Test {};

TEST_F(RefreshExternalCollectionJobInfoTest, SetterAndGetter) {
    milvus::RefreshExternalCollectionJobInfo info;
    info.SetJobID(101);
    info.SetCollectionName("coll");
    info.SetState(milvus::RefreshExternalCollectionStateCode::IN_PROGRESS);
    info.SetProgress(55);
    info.SetReason("running");
    info.SetExternalSource("s3://bucket/path/");
    info.SetStartTime(1000);
    info.SetEndTime(2000);

    EXPECT_EQ(info.JobID(), 101);
    EXPECT_EQ(info.CollectionName(), "coll");
    EXPECT_EQ(info.State(), milvus::RefreshExternalCollectionStateCode::IN_PROGRESS);
    EXPECT_EQ(info.Progress(), 55);
    EXPECT_EQ(info.Reason(), "running");
    EXPECT_EQ(info.ExternalSource(), "s3://bucket/path/");
    EXPECT_EQ(info.StartTime(), 1000);
    EXPECT_EQ(info.EndTime(), 2000);
}

TEST_F(RefreshExternalCollectionJobInfoTest, StateToString) {
    EXPECT_EQ(std::to_string(milvus::RefreshExternalCollectionStateCode::PENDING), "RefreshPending");
    EXPECT_EQ(std::to_string(milvus::RefreshExternalCollectionStateCode::IN_PROGRESS), "RefreshInProgress");
    EXPECT_EQ(std::to_string(milvus::RefreshExternalCollectionStateCode::COMPLETED), "RefreshCompleted");
    EXPECT_EQ(std::to_string(milvus::RefreshExternalCollectionStateCode::FAILED), "RefreshFailed");
}

class FileResourceInfoTest : public ::testing::Test {};

TEST_F(FileResourceInfoTest, SetterAndGetter) {
    milvus::FileResourceInfo info;
    info.SetName("res1");
    info.SetPath("/tmp/data.parquet");

    EXPECT_EQ(info.Name(), "res1");
    EXPECT_EQ(info.Path(), "/tmp/data.parquet");
}
