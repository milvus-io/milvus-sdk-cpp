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

#include <chrono>
#include <memory>
#include <thread>

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::GetReplicateConfigurationRequest;
using ::milvus::proto::milvus::GetReplicateConfigurationResponse;
using ::milvus::proto::milvus::GetReplicateInfoRequest;
using ::milvus::proto::milvus::GetReplicateInfoResponse;
using ::milvus::proto::milvus::UpdateReplicateConfigurationRequest;
using ::testing::_;

namespace {

std::shared_ptr<milvus::MilvusClientV2>
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
        .WillOnce(
            [](::grpc::ServerContext*, const GetReplicateConfigurationRequest*, GetReplicateConfigurationResponse*) {
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
        .WillOnce(
            [](::grpc::ServerContext*, const UpdateReplicateConfigurationRequest*, ::milvus::proto::common::Status*) {
                return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
            });

    milvus::UpdateReplicateConfigurationRequest request;
    request.WithConfiguration(CreateConfiguration());
    auto status = client->UpdateReplicateConfiguration(request);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, GetReplicateInfo) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetReplicateInfo(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const GetReplicateInfoRequest* request, GetReplicateInfoResponse* response) {
                EXPECT_EQ(request->source_cluster_id(), "cluster-a");
                EXPECT_EQ(request->target_pchannel(), "by-dev-rootcoord-dml_0");
                auto checkpoint = response->mutable_checkpoint();
                checkpoint->set_cluster_id("cluster-a");
                checkpoint->set_pchannel("by-dev-rootcoord-dml_0");
                checkpoint->set_time_tick(123);
                auto message_id = checkpoint->mutable_message_id();
                message_id->set_id("message-id");
                message_id->set_wal_name(::milvus::proto::common::WALName::Pulsar);

                auto salvage_checkpoint = response->mutable_salvage_checkpoint();
                salvage_checkpoint->set_cluster_id("cluster-a");
                salvage_checkpoint->set_pchannel("by-dev-rootcoord-dml_1");
                salvage_checkpoint->set_time_tick(456);
                auto salvage_message_id = salvage_checkpoint->mutable_message_id();
                salvage_message_id->set_id("salvage-message-id");
                salvage_message_id->set_wal_name(::milvus::proto::common::WALName::RocksMQ);
                return ::grpc::Status{};
            });

    milvus::GetReplicateInfoRequest request;
    request.WithSourceClusterID("cluster-a").WithTargetPChannel("by-dev-rootcoord-dml_0");
    milvus::GetReplicateInfoResponse response;
    auto status = client->GetReplicateInfo(request, response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.Checkpoint().ClusterID(), "cluster-a");
    EXPECT_EQ(response.Checkpoint().PChannel(), "by-dev-rootcoord-dml_0");
    EXPECT_EQ(response.Checkpoint().TimeTick(), 123);
    EXPECT_EQ(response.Checkpoint().MessageID().ID(), "message-id");
    EXPECT_EQ(response.Checkpoint().MessageID().WalName(), "Pulsar");

    EXPECT_EQ(response.SalvageCheckpoint().ClusterID(), "cluster-a");
    EXPECT_EQ(response.SalvageCheckpoint().PChannel(), "by-dev-rootcoord-dml_1");
    EXPECT_EQ(response.SalvageCheckpoint().TimeTick(), 456);
    EXPECT_EQ(response.SalvageCheckpoint().MessageID().ID(), "salvage-message-id");
    EXPECT_EQ(response.SalvageCheckpoint().MessageID().WalName(), "RocksMQ");
}

TEST_F(UnconnectMilvusMockedTest, GetReplicateInfoFailed) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetReplicateInfo(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetReplicateInfoRequest*, GetReplicateInfoResponse*) {
            return ::grpc::Status{::grpc::StatusCode::UNAVAILABLE, "unavailable"};
        });

    milvus::GetReplicateInfoRequest request;
    request.WithSourceClusterID("cluster-a").WithTargetPChannel("by-dev-rootcoord-dml_0");
    milvus::GetReplicateInfoResponse response;
    auto status = client->GetReplicateInfo(request, response);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::RPC_FAILED);
}

TEST_F(UnconnectMilvusMockedTest, GetReplicateInfoClearsCheckpointWhenAbsent) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, GetReplicateInfo(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const GetReplicateInfoRequest*, GetReplicateInfoResponse* response) {
            auto checkpoint = response->mutable_checkpoint();
            checkpoint->set_cluster_id("cluster-a");
            checkpoint->set_pchannel("by-dev-rootcoord-dml_0");
            checkpoint->set_time_tick(123);
            auto message_id = checkpoint->mutable_message_id();
            message_id->set_id("message-id");
            message_id->set_wal_name(::milvus::proto::common::WALName::Pulsar);

            auto salvage_checkpoint = response->mutable_salvage_checkpoint();
            salvage_checkpoint->set_cluster_id("cluster-a");
            salvage_checkpoint->set_pchannel("by-dev-rootcoord-dml_1");
            salvage_checkpoint->set_time_tick(456);
            auto salvage_message_id = salvage_checkpoint->mutable_message_id();
            salvage_message_id->set_id("salvage-message-id");
            salvage_message_id->set_wal_name(::milvus::proto::common::WALName::RocksMQ);
            return ::grpc::Status{};
        })
        .WillOnce([](::grpc::ServerContext*, const GetReplicateInfoRequest*, GetReplicateInfoResponse*) {
            return ::grpc::Status{};
        });

    milvus::GetReplicateInfoRequest request;
    request.WithSourceClusterID("cluster-a").WithTargetPChannel("by-dev-rootcoord-dml_0");
    milvus::GetReplicateInfoResponse response;
    auto status = client->GetReplicateInfo(request, response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.Checkpoint().ClusterID(), "cluster-a");
    EXPECT_EQ(response.Checkpoint().MessageID().ID(), "message-id");
    EXPECT_EQ(response.SalvageCheckpoint().ClusterID(), "cluster-a");
    EXPECT_EQ(response.SalvageCheckpoint().MessageID().ID(), "salvage-message-id");

    status = client->GetReplicateInfo(request, response);
    EXPECT_TRUE(status.IsOk());
    EXPECT_TRUE(response.Checkpoint().ClusterID().empty());
    EXPECT_TRUE(response.Checkpoint().PChannel().empty());
    EXPECT_TRUE(response.Checkpoint().MessageID().ID().empty());
    EXPECT_TRUE(response.Checkpoint().MessageID().WalName().empty());
    EXPECT_EQ(response.Checkpoint().TimeTick(), 0);
    EXPECT_TRUE(response.SalvageCheckpoint().ClusterID().empty());
    EXPECT_TRUE(response.SalvageCheckpoint().PChannel().empty());
    EXPECT_TRUE(response.SalvageCheckpoint().MessageID().ID().empty());
    EXPECT_TRUE(response.SalvageCheckpoint().MessageID().WalName().empty());
    EXPECT_EQ(response.SalvageCheckpoint().TimeTick(), 0);
}

TEST_F(UnconnectMilvusMockedTest, DumpMessagesWithoutConnect) {
    auto client = milvus::MilvusClientV2::Create();
    bool callback_called = false;

    milvus::DumpMessagesRequest request;
    request.WithPChannel("by-dev-rootcoord-dml_0");
    auto status = client->DumpMessages(request, [&callback_called](const milvus::DumpedMessage&) {
        callback_called = true;
        return milvus::Status::OK();
    });

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::NOT_CONNECTED);
    EXPECT_FALSE(callback_called);
}

TEST_F(UnconnectMilvusMockedTest, DumpMessagesRejectsInvalidWalName) {
    auto client = milvus::MilvusClientV2::Create();
    bool callback_called = false;

    milvus::ReplicateMessageID start_message_id;
    start_message_id.WithID("message-id").WithWalName("Rocksmq");

    milvus::DumpMessagesRequest request;
    request.WithPChannel("by-dev-rootcoord-dml_0").WithStartMessageID(std::move(start_message_id));
    auto status = client->DumpMessages(request, [&callback_called](const milvus::DumpedMessage&) {
        callback_called = true;
        return milvus::Status::OK();
    });

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "Unknown WAL name: Rocksmq");
    EXPECT_FALSE(callback_called);
}

TEST_F(UnconnectMilvusMockedTest, DumpMessagesRejectsEmptyCallback) {
    auto client = milvus::MilvusClientV2::Create();

    milvus::DumpMessagesRequest request;
    request.WithPChannel("by-dev-rootcoord-dml_0");
    std::function<milvus::Status(const milvus::DumpedMessage&)> callback;
    auto status = client->DumpMessages(request, callback);

    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "DumpMessages callback cannot be empty");
}

TEST_F(UnconnectMilvusMockedTest, DumpMessagesIgnoresGlobalRpcDeadline) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());
    auto status = client->SetRpcDeadlineMs(1);
    EXPECT_TRUE(status.IsOk());

    EXPECT_CALL(service_, DumpMessages(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ::milvus::proto::milvus::DumpMessagesRequest* request,
                     ::grpc::ServerWriter<::milvus::proto::milvus::DumpMessagesResponse>* writer) {
            EXPECT_EQ(request->pchannel(), "by-dev-rootcoord-dml_0");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));

            ::milvus::proto::milvus::DumpMessagesResponse response;
            auto* message = response.mutable_message();
            message->mutable_id()->set_id("message-id");
            message->mutable_id()->set_wal_name(::milvus::proto::common::WALName::RocksMQ);
            message->set_payload("payload");
            writer->Write(response);
            return ::grpc::Status{};
        });

    milvus::DumpMessagesRequest request;
    request.WithPChannel("by-dev-rootcoord-dml_0");

    std::vector<std::string> message_ids;
    status = client->DumpMessages(request, [&message_ids](const milvus::DumpedMessage& message) {
        message_ids.push_back(message.MessageID().ID());
        return milvus::Status::OK();
    });

    EXPECT_TRUE(status.IsOk());
    ASSERT_EQ(message_ids.size(), 1);
    EXPECT_EQ(message_ids[0], "message-id");
}
