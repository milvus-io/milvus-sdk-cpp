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

#include <utility>

#include "milvus/MilvusClientV2.h"

class GetReplicateConfigurationResponseTest : public ::testing::Test {};

TEST_F(GetReplicateConfigurationResponseTest, SetterAndGetter) {
    milvus::GetReplicateConfigurationResponse response;
    milvus::ReplicateConfiguration configuration;
    milvus::MilvusCluster cluster;
    cluster.WithClusterID("cluster-a");
    configuration.AddCluster(std::move(cluster));

    response.SetConfiguration(std::move(configuration));

    ASSERT_EQ(response.Configuration().Clusters().size(), 1u);
    EXPECT_EQ(response.Configuration().Clusters()[0].ClusterID(), "cluster-a");
}

class GetReplicateInfoResponseTest : public ::testing::Test {};

TEST_F(GetReplicateInfoResponseTest, SetterAndGetter) {
    milvus::GetReplicateInfoResponse response;
    milvus::ReplicateMessageID message_id;
    message_id.WithID("message-id").WithWalName("Pulsar");
    milvus::ReplicateCheckpoint checkpoint;
    checkpoint.WithClusterID("cluster-a")
        .WithPChannel("by-dev-rootcoord-dml_0")
        .WithMessageID(std::move(message_id))
        .WithTimeTick(123);

    milvus::ReplicateMessageID salvage_message_id;
    salvage_message_id.WithID("salvage-message-id").WithWalName("RocksMQ");
    milvus::ReplicateCheckpoint salvage_checkpoint;
    salvage_checkpoint.WithClusterID("cluster-a")
        .WithPChannel("by-dev-rootcoord-dml_1")
        .WithMessageID(std::move(salvage_message_id))
        .WithTimeTick(456);

    response.SetCheckpoint(std::move(checkpoint));
    response.SetSalvageCheckpoint(std::move(salvage_checkpoint));

    EXPECT_EQ(response.Checkpoint().ClusterID(), "cluster-a");
    EXPECT_EQ(response.Checkpoint().PChannel(), "by-dev-rootcoord-dml_0");
    EXPECT_EQ(response.Checkpoint().MessageID().ID(), "message-id");
    EXPECT_EQ(response.Checkpoint().MessageID().WalName(), "Pulsar");
    EXPECT_EQ(response.Checkpoint().TimeTick(), 123);

    EXPECT_EQ(response.SalvageCheckpoint().ClusterID(), "cluster-a");
    EXPECT_EQ(response.SalvageCheckpoint().PChannel(), "by-dev-rootcoord-dml_1");
    EXPECT_EQ(response.SalvageCheckpoint().MessageID().ID(), "salvage-message-id");
    EXPECT_EQ(response.SalvageCheckpoint().MessageID().WalName(), "RocksMQ");
    EXPECT_EQ(response.SalvageCheckpoint().TimeTick(), 456);
}

TEST_F(GetReplicateInfoResponseTest, SalvageCheckpointDefaultsEmpty) {
    milvus::GetReplicateInfoResponse response;

    EXPECT_TRUE(response.SalvageCheckpoint().ClusterID().empty());
    EXPECT_TRUE(response.SalvageCheckpoint().PChannel().empty());
    EXPECT_TRUE(response.SalvageCheckpoint().MessageID().ID().empty());
    EXPECT_TRUE(response.SalvageCheckpoint().MessageID().WalName().empty());
    EXPECT_EQ(response.SalvageCheckpoint().TimeTick(), 0);
}

class DumpedMessageTest : public ::testing::Test {};

TEST_F(DumpedMessageTest, GettersSettersAndFluentMethods) {
    milvus::DumpedMessage message;
    EXPECT_TRUE(message.MessageID().ID().empty());
    EXPECT_TRUE(message.Payload().empty());
    EXPECT_TRUE(message.Properties().empty());

    milvus::ReplicateMessageID message_id;
    message_id.WithID("message-id").WithWalName("Pulsar");
    message.SetMessageID(std::move(message_id));
    message.SetPayload("payload");
    std::unordered_map<std::string, std::string> properties{{"k1", "v1"}};
    message.SetProperties(std::move(properties));
    EXPECT_EQ(message.MessageID().ID(), "message-id");
    EXPECT_EQ(message.MessageID().WalName(), "Pulsar");
    EXPECT_EQ(message.Payload(), "payload");
    ASSERT_EQ(message.Properties().size(), 1u);
    EXPECT_EQ(message.Properties().at("k1"), "v1");

    milvus::ReplicateMessageID next_message_id;
    next_message_id.WithID("message-id-2").WithWalName("Rocksmq");
    std::unordered_map<std::string, std::string> next_properties{{"k2", "v2"}};
    auto& ref = message.WithMessageID(std::move(next_message_id))
                    .WithPayload("payload-2")
                    .WithProperties(std::move(next_properties))
                    .AddProperty("k3", "v3");
    EXPECT_EQ(&ref, &message);
    EXPECT_EQ(message.MessageID().ID(), "message-id-2");
    EXPECT_EQ(message.MessageID().WalName(), "Rocksmq");
    EXPECT_EQ(message.Payload(), "payload-2");
    ASSERT_EQ(message.Properties().size(), 2u);
    EXPECT_EQ(message.Properties().at("k2"), "v2");
    EXPECT_EQ(message.Properties().at("k3"), "v3");
}
