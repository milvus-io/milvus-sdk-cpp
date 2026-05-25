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

class UpdateReplicateConfigurationRequestTest : public ::testing::Test {};

TEST_F(UpdateReplicateConfigurationRequestTest, GettersSettersAndFluentMethods) {
    milvus::UpdateReplicateConfigurationRequest request;
    EXPECT_FALSE(request.ForcePromote());

    milvus::ReplicateConfiguration configuration;
    milvus::MilvusCluster cluster;
    cluster.WithClusterID("cluster-a");
    configuration.AddCluster(std::move(cluster));
    request.SetConfiguration(std::move(configuration));
    EXPECT_EQ(request.Configuration().Clusters().size(), 1);
    EXPECT_EQ(request.Configuration().Clusters()[0].ClusterID(), "cluster-a");

    milvus::ReplicateConfiguration new_configuration;
    milvus::MilvusCluster new_cluster;
    new_cluster.WithClusterID("cluster-b");
    new_configuration.AddCluster(std::move(new_cluster));
    auto& ref = request.WithConfiguration(std::move(new_configuration)).WithForcePromote(true);
    EXPECT_EQ(&ref, &request);
    EXPECT_TRUE(request.ForcePromote());
    EXPECT_EQ(request.Configuration().Clusters().size(), 1);
    EXPECT_EQ(request.Configuration().Clusters()[0].ClusterID(), "cluster-b");
}

class GetReplicateConfigurationRequestTest : public ::testing::Test {};

TEST_F(GetReplicateConfigurationRequestTest, DefaultConstructible) {
    milvus::GetReplicateConfigurationRequest request;
    (void)request;
}
