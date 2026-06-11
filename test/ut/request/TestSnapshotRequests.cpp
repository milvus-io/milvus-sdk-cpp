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

class CreateSnapshotRequestTest : public ::testing::Test {};

TEST_F(CreateSnapshotRequestTest, GettersAndSetters) {
    milvus::CreateSnapshotRequest req;
    req.WithDatabaseName("db")
        .WithCollectionName("coll")
        .WithSnapshotName("snap")
        .WithDescription("desc")
        .WithCompactionProtectionSeconds(123);

    EXPECT_EQ(req.DatabaseName(), "db");
    EXPECT_EQ(req.CollectionName(), "coll");
    EXPECT_EQ(req.SnapshotName(), "snap");
    EXPECT_EQ(req.Description(), "desc");
    EXPECT_EQ(req.CompactionProtectionSeconds(), 123);
}

class DropSnapshotRequestTest : public ::testing::Test {};

TEST_F(DropSnapshotRequestTest, GettersAndSetters) {
    milvus::DropSnapshotRequest req;
    req.WithDatabaseName("db").WithCollectionName("coll").WithSnapshotName("snap");

    EXPECT_EQ(req.DatabaseName(), "db");
    EXPECT_EQ(req.CollectionName(), "coll");
    EXPECT_EQ(req.SnapshotName(), "snap");
}

class ListSnapshotsRequestTest : public ::testing::Test {};

TEST_F(ListSnapshotsRequestTest, GettersAndSetters) {
    milvus::ListSnapshotsRequest req;
    req.WithDatabaseName("db").WithCollectionName("coll");

    EXPECT_EQ(req.DatabaseName(), "db");
    EXPECT_EQ(req.CollectionName(), "coll");
}

class DescribeSnapshotRequestTest : public ::testing::Test {};

TEST_F(DescribeSnapshotRequestTest, GettersAndSetters) {
    milvus::DescribeSnapshotRequest req;
    req.WithDatabaseName("db").WithCollectionName("coll").WithSnapshotName("snap");

    EXPECT_EQ(req.DatabaseName(), "db");
    EXPECT_EQ(req.CollectionName(), "coll");
    EXPECT_EQ(req.SnapshotName(), "snap");
}

class RestoreSnapshotRequestTest : public ::testing::Test {};

TEST_F(RestoreSnapshotRequestTest, GettersAndSetters) {
    milvus::RestoreSnapshotRequest req;
    req.WithSnapshotName("snap")
        .WithSourceDatabaseName("source_db")
        .WithSourceCollectionName("source_coll")
        .WithTargetDatabaseName("target_db")
        .WithTargetCollectionName("target_coll");

    EXPECT_EQ(req.SnapshotName(), "snap");
    EXPECT_EQ(req.SourceDatabaseName(), "source_db");
    EXPECT_EQ(req.SourceCollectionName(), "source_coll");
    EXPECT_EQ(req.TargetDatabaseName(), "target_db");
    EXPECT_EQ(req.TargetCollectionName(), "target_coll");
}

class GetRestoreSnapshotStateRequestTest : public ::testing::Test {};

TEST_F(GetRestoreSnapshotStateRequestTest, GettersAndSetters) {
    milvus::GetRestoreSnapshotStateRequest req;
    req.WithJobID(101);
    EXPECT_EQ(req.JobID(), 101);
}

class ListRestoreSnapshotJobsRequestTest : public ::testing::Test {};

TEST_F(ListRestoreSnapshotJobsRequestTest, GettersAndSetters) {
    milvus::ListRestoreSnapshotJobsRequest req;
    req.WithDatabaseName("db").WithCollectionName("coll");

    EXPECT_EQ(req.DatabaseName(), "db");
    EXPECT_EQ(req.CollectionName(), "coll");
}

class PinSnapshotDataRequestTest : public ::testing::Test {};

TEST_F(PinSnapshotDataRequestTest, GettersAndSetters) {
    milvus::PinSnapshotDataRequest req;
    req.WithDatabaseName("db").WithCollectionName("coll").WithSnapshotName("snap").WithTtlSeconds(60);

    EXPECT_EQ(req.DatabaseName(), "db");
    EXPECT_EQ(req.CollectionName(), "coll");
    EXPECT_EQ(req.SnapshotName(), "snap");
    EXPECT_EQ(req.TtlSeconds(), 60);
}

class UnpinSnapshotDataRequestTest : public ::testing::Test {};

TEST_F(UnpinSnapshotDataRequestTest, GettersAndSetters) {
    milvus::UnpinSnapshotDataRequest req;
    req.WithPinID(555);
    EXPECT_EQ(req.PinID(), 555);
}
