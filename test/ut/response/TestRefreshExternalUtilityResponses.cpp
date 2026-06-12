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

class RefreshExternalCollectionResponseTest : public ::testing::Test {};

TEST_F(RefreshExternalCollectionResponseTest, SetterAndGetter) {
    milvus::RefreshExternalCollectionResponse resp;
    resp.SetJobID(88);
    EXPECT_EQ(resp.JobID(), 88);
}

class GetRefreshExternalCollectionProgressResponseTest : public ::testing::Test {};

TEST_F(GetRefreshExternalCollectionProgressResponseTest, SetterAndGetter) {
    milvus::GetRefreshExternalCollectionProgressResponse resp;
    milvus::RefreshExternalCollectionJobInfo info;
    info.SetJobID(101);
    resp.SetJobInfo(std::move(info));
    EXPECT_EQ(resp.JobInfo().JobID(), 101);
}

class ListRefreshExternalCollectionJobsResponseTest : public ::testing::Test {};

TEST_F(ListRefreshExternalCollectionJobsResponseTest, SetterAndGetter) {
    milvus::ListRefreshExternalCollectionJobsResponse resp;
    std::vector<milvus::RefreshExternalCollectionJobInfo> jobs;
    milvus::RefreshExternalCollectionJobInfo info;
    info.SetJobID(11);
    jobs.push_back(std::move(info));
    resp.SetJobs(std::move(jobs));
    ASSERT_EQ(resp.Jobs().size(), 1);
    EXPECT_EQ(resp.Jobs()[0].JobID(), 11);
}

class ListFileResourcesResponseTest : public ::testing::Test {};

TEST_F(ListFileResourcesResponseTest, SetterAndGetter) {
    milvus::ListFileResourcesResponse resp;
    std::vector<milvus::FileResourceInfo> resources;
    milvus::FileResourceInfo info;
    info.SetName("res1");
    info.SetPath("/tmp/data.parquet");
    resources.push_back(std::move(info));
    resp.SetResources(std::move(resources));
    ASSERT_EQ(resp.Resources().size(), 1);
    EXPECT_EQ(resp.Resources()[0].Name(), "res1");
    EXPECT_EQ(resp.Resources()[0].Path(), "/tmp/data.parquet");
}
