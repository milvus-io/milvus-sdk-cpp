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

class RefreshExternalCollectionRequestTest : public ::testing::Test {};

TEST_F(RefreshExternalCollectionRequestTest, GettersAndSetters) {
    milvus::RefreshExternalCollectionRequest req;
    req.WithDatabaseName("db")
        .WithCollectionName("coll")
        .WithExternalSource("s3://bucket/path/")
        .WithExternalSpec(nlohmann::json{{"format", "parquet"}});

    EXPECT_EQ(req.DatabaseName(), "db");
    EXPECT_EQ(req.CollectionName(), "coll");
    EXPECT_EQ(req.ExternalSource(), "s3://bucket/path/");
    EXPECT_EQ(req.ExternalSpec().at("format"), "parquet");
}

class GetRefreshExternalCollectionProgressRequestTest : public ::testing::Test {};

TEST_F(GetRefreshExternalCollectionProgressRequestTest, GettersAndSetters) {
    milvus::GetRefreshExternalCollectionProgressRequest req;
    req.WithJobID(123);
    EXPECT_EQ(req.JobID(), 123);
}

class ListRefreshExternalCollectionJobsRequestTest : public ::testing::Test {};

TEST_F(ListRefreshExternalCollectionJobsRequestTest, GettersAndSetters) {
    milvus::ListRefreshExternalCollectionJobsRequest req;
    req.WithDatabaseName("db").WithCollectionName("coll");
    EXPECT_EQ(req.DatabaseName(), "db");
    EXPECT_EQ(req.CollectionName(), "coll");
}

class AddFileResourceRequestTest : public ::testing::Test {};

TEST_F(AddFileResourceRequestTest, GettersAndSetters) {
    milvus::AddFileResourceRequest req;
    req.WithName("res1").WithPath("/tmp/data.parquet");
    EXPECT_EQ(req.Name(), "res1");
    EXPECT_EQ(req.Path(), "/tmp/data.parquet");
}

class RemoveFileResourceRequestTest : public ::testing::Test {};

TEST_F(RemoveFileResourceRequestTest, GettersAndSetters) {
    milvus::RemoveFileResourceRequest req;
    req.WithName("res1");
    EXPECT_EQ(req.Name(), "res1");
}

class ListFileResourcesRequestTest : public ::testing::Test {};

TEST_F(ListFileResourcesRequestTest, DefaultConstruction) {
    milvus::ListFileResourcesRequest req;
    (void)req;
}
