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

#include <iostream>
#include <string>
#include <vector>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {

const std::string kClusterAUri = "http://localhost:19530";
const std::string kClusterBUri = "http://localhost:29530";
const std::string kClusterAId = "cdc-a";
const std::string kClusterBId = "cdc-b";
const int kPChannelNum = 16;

void
CheckStatus(const std::string& message, const milvus::Status& status) {
    util::CheckStatus(std::string(message), status);
}

std::vector<std::string>
GeneratePChannels(const std::string& cluster_id) {
    std::vector<std::string> pchannels;
    pchannels.reserve(kPChannelNum);
    for (int i = 0; i < kPChannelNum; ++i) {
        pchannels.push_back(cluster_id + "-rootcoord-dml_" + std::to_string(i));
    }
    return pchannels;
}

void
PrintReplicateConfiguration(const milvus::ReplicateConfiguration& configuration) {
    std::cout << "Replicate configuration:" << std::endl;
    for (const auto& cluster : configuration.Clusters()) {
        std::cout << "  clusterId=" << cluster.ClusterID() << ", uri=" << cluster.Uri() << ", pchannels=[";
        for (size_t i = 0; i < cluster.PChannels().size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << cluster.PChannels()[i];
        }
        std::cout << "]" << std::endl;
    }
    for (const auto& topology : configuration.CrossClusterTopologies()) {
        std::cout << "  topology: sourceClusterId=" << topology.SourceClusterID()
                  << ", targetClusterId=" << topology.TargetClusterID() << std::endl;
    }
}

void
PrintCheckpoint(const std::string& name, const milvus::ReplicateCheckpoint& checkpoint) {
    std::cout << "  " << name << ": clusterId=" << checkpoint.ClusterID() << ", pchannel=" << checkpoint.PChannel()
              << ", messageId=" << checkpoint.MessageID().ID() << ", walName=" << checkpoint.MessageID().WalName()
              << ", timeTick=" << checkpoint.TimeTick() << std::endl;
}

void
PrintReplicateInfo(const milvus::GetReplicateInfoResponse& response) {
    std::cout << "Replicate info:" << std::endl;
    PrintCheckpoint("checkpoint", response.Checkpoint());
    PrintCheckpoint("salvageCheckpoint", response.SalvageCheckpoint());
}

void
PrintDumpedMessage(const milvus::DumpedMessage& message) {
    std::cout << "\tmessage id: " << message.MessageID().ID() << std::endl;
    std::cout << "\tpayload size: " << message.Payload().size() << std::endl;
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto cluster_a_client = milvus::MilvusClientV2::Create();
    auto cluster_b_client = milvus::MilvusClientV2::Create();

    auto status = cluster_a_client->Connect(milvus::ConnectParam{kClusterAUri});
    CheckStatus("connect cluster A", status);
    std::string cluster_a_version;
    status = cluster_a_client->GetServerVersion(cluster_a_version);
    CheckStatus("get cluster A server version", status);
    std::cout << "Cluster A connected: " << cluster_a_version << std::endl;

    status = cluster_b_client->Connect(milvus::ConnectParam{kClusterBUri});
    CheckStatus("connect cluster B", status);
    std::string cluster_b_version;
    status = cluster_b_client->GetServerVersion(cluster_b_version);
    CheckStatus("get cluster B server version", status);
    std::cout << "Cluster B connected: " << cluster_b_version << std::endl;

    milvus::MilvusCluster cluster_a;
    cluster_a.WithClusterID(kClusterAId).WithUri(kClusterAUri).WithPChannels(GeneratePChannels(kClusterAId));

    milvus::MilvusCluster cluster_b;
    cluster_b.WithClusterID(kClusterBId).WithUri(kClusterBUri).WithPChannels(GeneratePChannels(kClusterBId));

    milvus::CrossClusterTopology topology;
    topology.WithSourceClusterID(kClusterAId).WithTargetClusterID(kClusterBId);

    milvus::ReplicateConfiguration configuration;
    configuration.AddCluster(std::move(cluster_a));
    configuration.AddCluster(std::move(cluster_b));
    configuration.AddCrossClusterTopology(std::move(topology));

    milvus::UpdateReplicateConfigurationRequest update_request;
    update_request.WithConfiguration(std::move(configuration));

    status = cluster_a_client->UpdateReplicateConfiguration(update_request);
    CheckStatus("update replicate configuration on cluster A", status);
    std::cout << "Replicate configuration updated for cluster A" << std::endl;

    status = cluster_b_client->UpdateReplicateConfiguration(update_request);
    CheckStatus("update replicate configuration on cluster B", status);
    std::cout << "Replicate configuration updated for cluster B" << std::endl;

    milvus::GetReplicateConfigurationResponse replicate_configuration_response;
    status = cluster_a_client->GetReplicateConfiguration(milvus::GetReplicateConfigurationRequest{},
                                                         replicate_configuration_response);
    CheckStatus("get replicate configuration", status);
    PrintReplicateConfiguration(replicate_configuration_response.Configuration());

    milvus::GetReplicateInfoRequest replicate_info_request;
    replicate_info_request.WithSourceClusterID(kClusterAId).WithTargetPChannel(GeneratePChannels(kClusterBId).front());

    milvus::GetReplicateInfoResponse replicate_info_response;
    status = cluster_b_client->GetReplicateInfo(replicate_info_request, replicate_info_response);
    CheckStatus("get replicate info from cluster B", status);
    PrintReplicateInfo(replicate_info_response);

    const std::string dump_start_message_id = replicate_info_response.SalvageCheckpoint().MessageID().ID();

    milvus::ReplicateMessageID start_message_id;
    start_message_id.WithID(dump_start_message_id).WithWalName("RocksMQ");

    milvus::DumpMessagesRequest dump_request;
    dump_request.WithPChannel(GeneratePChannels(kClusterAId).front()).WithStartMessageID(std::move(start_message_id));

    std::cout << "Dump messages:" << std::endl;
    status = cluster_a_client->DumpMessages(dump_request, [](const milvus::DumpedMessage& message) {
        PrintDumpedMessage(message);
        return milvus::Status::OK();
    });
    CheckStatus("dump messages from cluster A", status);

    cluster_a_client->Disconnect();
    cluster_b_client->Disconnect();
    return 0;
}
