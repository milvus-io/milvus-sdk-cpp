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

class ListSnapshotsResponseTest : public ::testing::Test {};

TEST_F(ListSnapshotsResponseTest, SetterAndGetter) {
    milvus::ListSnapshotsResponse resp;
    std::vector<std::string> snapshots{"snap1", "snap2"};
    resp.SetSnapshots(std::move(snapshots));
    ASSERT_EQ(resp.Snapshots().size(), 2);
    EXPECT_EQ(resp.Snapshots()[0], "snap1");
    EXPECT_EQ(resp.Snapshots()[1], "snap2");
}

class DescribeSnapshotResponseTest : public ::testing::Test {};

TEST_F(DescribeSnapshotResponseTest, SetterAndGetter) {
    milvus::DescribeSnapshotResponse resp;
    resp.SetName("snap");
    resp.SetDescription("desc");
    resp.SetCollectionName("coll");
    std::vector<std::string> partitions{"p1", "p2"};
    resp.SetPartitionNames(std::move(partitions));
    resp.SetCreateTs(1234);
    resp.SetS3Location("s3://bucket/meta");

    EXPECT_EQ(resp.Name(), "snap");
    EXPECT_EQ(resp.Description(), "desc");
    EXPECT_EQ(resp.CollectionName(), "coll");
    ASSERT_EQ(resp.PartitionNames().size(), 2);
    EXPECT_EQ(resp.PartitionNames()[0], "p1");
    EXPECT_EQ(resp.CreateTs(), 1234);
    EXPECT_EQ(resp.S3Location(), "s3://bucket/meta");
}

class RestoreSnapshotResponseTest : public ::testing::Test {};

TEST_F(RestoreSnapshotResponseTest, SetterAndGetter) {
    milvus::RestoreSnapshotResponse resp;
    resp.SetJobID(321);
    EXPECT_EQ(resp.JobID(), 321);
}

class GetRestoreSnapshotStateResponseTest : public ::testing::Test {};

TEST_F(GetRestoreSnapshotStateResponseTest, SetterAndGetter) {
    milvus::GetRestoreSnapshotStateResponse resp;
    milvus::RestoreSnapshotJobInfo info;
    info.SetJobID(10);
    resp.SetJobInfo(std::move(info));
    EXPECT_EQ(resp.JobInfo().JobID(), 10);
}

class ListRestoreSnapshotJobsResponseTest : public ::testing::Test {};

TEST_F(ListRestoreSnapshotJobsResponseTest, SetterAndGetter) {
    milvus::ListRestoreSnapshotJobsResponse resp;
    std::vector<milvus::RestoreSnapshotJobInfo> jobs;
    milvus::RestoreSnapshotJobInfo info;
    info.SetJobID(11);
    jobs.push_back(std::move(info));
    resp.SetJobs(std::move(jobs));
    ASSERT_EQ(resp.Jobs().size(), 1);
    EXPECT_EQ(resp.Jobs()[0].JobID(), 11);
}

class PinSnapshotDataResponseTest : public ::testing::Test {};

TEST_F(PinSnapshotDataResponseTest, SetterAndGetter) {
    milvus::PinSnapshotDataResponse resp;
    resp.SetPinID(88);
    EXPECT_EQ(resp.PinID(), 88);
}
