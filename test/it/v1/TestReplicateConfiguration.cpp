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

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::GetReplicateConfigurationRequest;
using ::milvus::proto::milvus::GetReplicateConfigurationResponse;
using ::milvus::proto::milvus::UpdateReplicateConfigurationRequest;
using ::testing::_;

namespace {

milvus::MilvusClientV2Ptr
CreateConnectedV2Client(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{"127.0.0.1", port};
    auto status = client->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());
    return client;
}

void
ExpectClusterEquals(const milvus::MilvusCluster& cluster, const milvus::proto::common::MilvusCluster& rpc_cluster) {
    EXPECT_EQ(rpc_cluster.cluster_id(), cluster.ClusterID());
    EXPECT_EQ(rpc_cluster.connection_param().uri(), cluster.Uri());
    EXPECT_EQ(rpc_cluster.connection_param().token(), cluster.Token());
    ASSERT_EQ(rpc_cluster.pchannels_size(), cluster.PChannels().size());
    for (int i = 0; i < rpc_cluster.pchannels_size(); ++i) {
        EXPECT_EQ(rpc_cluster.pchannels(i), cluster.PChannels()[i]);
    }
}

void
ExpectTopologyEquals(const milvus::CrossClusterTopology& topology,
                     const milvus::proto::common::CrossClusterTopology& rpc_topology) {
    EXPECT_EQ(rpc_topology.source_cluster_id(), topology.SourceClusterID());
    EXPECT_EQ(rpc_topology.target_cluster_id(), topology.TargetClusterID());
}

milvus::ReplicateConfiguration
CreateConfiguration() {
    milvus::ReplicateConfiguration configuration;
    milvus::MilvusCluster cluster;
    cluster.WithClusterID("cluster-a")
        .WithUri("http://localhost:19530")
        .WithToken("token-a")
        .WithPChannels({"by-dev-rootcoord-dml_0", "by-dev-rootcoord-dml_1"});
    configuration.AddCluster(std::move(cluster));

    milvus::CrossClusterTopology topology;
    topology.WithSourceClusterID("cluster-a").WithTargetClusterID("cluster-b");
    configuration.AddCrossClusterTopology(std::move(topology));
    return configuration;
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, GetReplicateConfiguration) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetReplicateConfiguration(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetReplicateConfigurationRequest*,
                     GetReplicateConfigurationResponse* response) {
            auto configuration = response->mutable_configuration();
            auto cluster = configuration->add_clusters();
            cluster->set_cluster_id("cluster-a");
            cluster->mutable_connection_param()->set_uri("http://localhost:19530");
            cluster->mutable_connection_param()->set_token("token-a");
            cluster->add_pchannels("by-dev-rootcoord-dml_0");
            cluster->add_pchannels("by-dev-rootcoord-dml_1");
            auto topology = configuration->add_cross_cluster_topology();
            topology->set_source_cluster_id("cluster-a");
            topology->set_target_cluster_id("cluster-b");
            return ::grpc::Status{};
        });

    milvus::GetReplicateConfigurationResponse response;
    auto status = client->GetReplicateConfiguration(milvus::GetReplicateConfigurationRequest{}, response);

    EXPECT_TRUE(status.IsOk());
    ASSERT_EQ(response.Configuration().Clusters().size(), 1);
    EXPECT_EQ(response.Configuration().Clusters()[0].ClusterID(), "cluster-a");
    EXPECT_EQ(response.Configuration().Clusters()[0].Uri(), "http://localhost:19530");
    EXPECT_EQ(response.Configuration().Clusters()[0].Token(), "token-a");
    ASSERT_EQ(response.Configuration().Clusters()[0].PChannels().size(), 2);
    EXPECT_EQ(response.Configuration().Clusters()[0].PChannels()[0], "by-dev-rootcoord-dml_0");
    ASSERT_EQ(response.Configuration().CrossClusterTopologies().size(), 1);
    EXPECT_EQ(response.Configuration().CrossClusterTopologies()[0].SourceClusterID(), "cluster-a");
    EXPECT_EQ(response.Configuration().CrossClusterTopologies()[0].TargetClusterID(), "cluster-b");
}

TEST_F(UnconnectMilvusMockedTest, GetReplicateConfigurationFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetReplicateConfiguration(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetReplicateConfigurationRequest*,
                     GetReplicateConfigurationResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::GetReplicateConfigurationResponse response;
    auto status = client->GetReplicateConfiguration(milvus::GetReplicateConfigurationRequest{}, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, UpdateReplicateConfiguration) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    auto configuration = CreateConfiguration();
    const auto expected_cluster = configuration.Clusters()[0];
    const auto expected_topology = configuration.CrossClusterTopologies()[0];

    EXPECT_CALL(service_, UpdateReplicateConfiguration(_, _, _))
        .WillOnce([&expected_cluster, &expected_topology](::grpc::ServerContext*,
                                                          const UpdateReplicateConfigurationRequest* request,
                                                          ::milvus::proto::common::Status*) {
            EXPECT_TRUE(request->force_promote());
            const auto& rpc_config = request->replicate_configuration();
            EXPECT_EQ(rpc_config.clusters_size(), 1);
            ExpectClusterEquals(expected_cluster, rpc_config.clusters(0));
            EXPECT_EQ(rpc_config.cross_cluster_topology_size(), 1);
            ExpectTopologyEquals(expected_topology, rpc_config.cross_cluster_topology(0));
            return ::grpc::Status{};
        });

    milvus::UpdateReplicateConfigurationRequest request;
    request.WithConfiguration(std::move(configuration)).WithForcePromote(true);
    auto status = client->UpdateReplicateConfiguration(request);

    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, UpdateReplicateConfigurationFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, UpdateReplicateConfiguration(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const UpdateReplicateConfigurationRequest*,
                     ::milvus::proto::common::Status*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::UpdateReplicateConfigurationRequest request;
    request.WithConfiguration(CreateConfiguration());
    auto status = client->UpdateReplicateConfiguration(request);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}
