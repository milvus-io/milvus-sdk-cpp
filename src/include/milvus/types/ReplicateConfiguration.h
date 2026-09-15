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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief A cluster participating in a replication configuration.
 */
class MILVUS_SDK_API MilvusCluster {
 public:
    /**
     * @brief Get the cluster identifier.
     * @return the cluster ID.
     */
    const std::string&
    ClusterID() const;

    /**
     * @brief Set the cluster identifier.
     * @param [in] cluster_id the cluster ID.
     */
    void
    SetClusterID(const std::string& cluster_id);

    /**
     * @brief Set the cluster identifier.
     * @param [in] cluster_id the cluster ID.
     */
    MilvusCluster&
    WithClusterID(const std::string& cluster_id);

    /**
     * @brief Get the cluster endpoint URI.
     * @return the URI.
     */
    const std::string&
    Uri() const;

    /**
     * @brief Set the cluster endpoint URI.
     * @param [in] uri the URI.
     */
    void
    SetUri(const std::string& uri);

    /**
     * @brief Set the cluster endpoint URI.
     * @param [in] uri the URI.
     */
    MilvusCluster&
    WithUri(const std::string& uri);

    /**
     * @brief Get the access token of the cluster.
     * @return the token.
     */
    const std::string&
    Token() const;

    /**
     * @brief Set the access token of the cluster.
     * @param [in] token the token.
     */
    void
    SetToken(const std::string& token);

    /**
     * @brief Set the access token of the cluster.
     * @param [in] token the token.
     */
    MilvusCluster&
    WithToken(const std::string& token);

    /**
     * @brief Get the Pulsar channels of the cluster.
     * @return the p channels.
     */
    const std::vector<std::string>&
    PChannels() const;

    /**
     * @brief Set the Pulsar channels of the cluster.
     * @param [in] pchannels the pchannels.
     */
    void
    SetPChannels(std::vector<std::string>&& pchannels);

    /**
     * @brief Set the Pulsar channels of the cluster.
     * @param [in] pchannels the pchannels.
     */
    MilvusCluster&
    WithPChannels(std::vector<std::string>&& pchannels);

    /**
     * @brief Add a Pulsar channel to the cluster.
     * @param [in] pchannel the pchannel.
     */
    MilvusCluster&
    AddPChannel(const std::string& pchannel);

 private:
    std::string cluster_id_;
    std::string uri_;
    std::string token_;
    std::vector<std::string> pchannels_;
};

/**
 * @brief A cross-cluster data forwarding edge between two clusters.
 */
class MILVUS_SDK_API CrossClusterTopology {
 public:
    /**
     * @brief Get the source cluster identifier.
     * @return the source cluster ID.
     */
    const std::string&
    SourceClusterID() const;

    /**
     * @brief Set the source cluster identifier.
     * @param [in] cluster_id the cluster ID.
     */
    void
    SetSourceClusterID(const std::string& cluster_id);

    /**
     * @brief Set the source cluster identifier.
     * @param [in] cluster_id the cluster ID.
     */
    CrossClusterTopology&
    WithSourceClusterID(const std::string& cluster_id);

    /**
     * @brief Get the target cluster identifier.
     * @return the target cluster ID.
     */
    const std::string&
    TargetClusterID() const;

    /**
     * @brief Set the target cluster identifier.
     * @param [in] cluster_id the cluster ID.
     */
    void
    SetTargetClusterID(const std::string& cluster_id);

    /**
     * @brief Set the target cluster identifier.
     * @param [in] cluster_id the cluster ID.
     */
    CrossClusterTopology&
    WithTargetClusterID(const std::string& cluster_id);

 private:
    std::string source_cluster_id_;
    std::string target_cluster_id_;
};

/**
 * @brief A message identifier in a Pulsar channel.
 */
class MILVUS_SDK_API ReplicateMessageID {
 public:
    /**
     * @brief Get the message identifier.
     * @return the ID.
     */
    const std::string&
    ID() const;

    /**
     * @brief Set the message identifier.
     * @param [in] id the ID.
     */
    void
    SetID(const std::string& id);

    /**
     * @brief Set the message identifier.
     * @param [in] id the ID.
     */
    ReplicateMessageID&
    WithID(const std::string& id);

    /**
     * @brief Get the WAL (write-ahead log) backend name of the message.
     * @return the wal name.
     */
    const std::string&
    WalName() const;

    /**
     * @brief Set the WAL (write-ahead log) backend name of the message.
     * @param [in] wal_name the wal name.
     */
    void
    SetWalName(const std::string& wal_name);

    /**
     * @brief Set the WAL (write-ahead log) backend name of the message.
     * @param [in] wal_name the wal name.
     */
    ReplicateMessageID&
    WithWalName(const std::string& wal_name);

 private:
    std::string id_;
    std::string wal_name_;
};

/**
 * @brief A replication checkpoint of a Pulsar channel in a cluster.
 */
class MILVUS_SDK_API ReplicateCheckpoint {
 public:
    /**
     * @brief Get the cluster identifier.
     * @return the cluster ID.
     */
    const std::string&
    ClusterID() const;

    /**
     * @brief Set the cluster identifier.
     * @param [in] cluster_id the cluster ID.
     */
    void
    SetClusterID(const std::string& cluster_id);

    /**
     * @brief Set the cluster identifier.
     * @param [in] cluster_id the cluster ID.
     */
    ReplicateCheckpoint&
    WithClusterID(const std::string& cluster_id);

    /**
     * @brief Get the Pulsar channel name.
     * @return the p channel.
     */
    const std::string&
    PChannel() const;

    /**
     * @brief Set the Pulsar channel name.
     * @param [in] pchannel the pchannel.
     */
    void
    SetPChannel(const std::string& pchannel);

    /**
     * @brief Set the Pulsar channel name.
     * @param [in] pchannel the pchannel.
     */
    ReplicateCheckpoint&
    WithPChannel(const std::string& pchannel);

    /**
     * @brief Get the message identifier of the checkpoint.
     * @return the message ID.
     */
    const ReplicateMessageID&
    MessageID() const;

    /**
     * @brief Set the message identifier of the checkpoint.
     * @param [in] message_id the message ID.
     */
    void
    SetMessageID(ReplicateMessageID&& message_id);

    /**
     * @brief Set the message identifier of the checkpoint.
     * @param [in] message_id the message ID.
     */
    ReplicateCheckpoint&
    WithMessageID(ReplicateMessageID&& message_id);

    /**
     * @brief Get the time tick of the checkpoint.
     * @return the time tick.
     */
    uint64_t
    TimeTick() const;

    /**
     * @brief Set the time tick of the checkpoint.
     * @param [in] time_tick the time tick.
     */
    void
    SetTimeTick(uint64_t time_tick);

    /**
     * @brief Set the time tick of the checkpoint.
     * @param [in] time_tick the time tick.
     */
    ReplicateCheckpoint&
    WithTimeTick(uint64_t time_tick);

 private:
    std::string cluster_id_;
    std::string pchannel_;
    ReplicateMessageID message_id_;
    uint64_t time_tick_{0};
};

/**
 * @brief Configuration of a replication topology, used by MilvusClientV2::UpdateReplicateConfiguration().
 */
class MILVUS_SDK_API ReplicateConfiguration {
 public:
    /**
     * @brief Get the clusters of the replication topology.
     * @return the clusters.
     */
    const std::vector<MilvusCluster>&
    Clusters() const;

    /**
     * @brief Set the clusters of the replication topology.
     * @param [in] clusters the clusters.
     */
    void
    SetClusters(std::vector<MilvusCluster>&& clusters);

    /**
     * @brief Set the clusters of the replication topology.
     * @param [in] clusters the clusters.
     */
    ReplicateConfiguration&
    WithClusters(std::vector<MilvusCluster>&& clusters);

    /**
     * @brief Add a cluster to the replication topology.
     * @param [in] cluster the cluster.
     */
    ReplicateConfiguration&
    AddCluster(MilvusCluster&& cluster);

    /**
     * @brief Get the cross-cluster forwarding edges of the topology.
     * @return the cross cluster topologies.
     */
    const std::vector<CrossClusterTopology>&
    CrossClusterTopologies() const;

    /**
     * @brief Set the cross-cluster forwarding edges of the topology.
     * @param [in] topologies the topologies.
     */
    void
    SetCrossClusterTopologies(std::vector<CrossClusterTopology>&& topologies);

    /**
     * @brief Set the cross-cluster forwarding edges of the topology.
     * @param [in] topologies the topologies.
     */
    ReplicateConfiguration&
    WithCrossClusterTopologies(std::vector<CrossClusterTopology>&& topologies);

    /**
     * @brief Add a cross-cluster forwarding edge to the topology.
     * @param [in] topology the topology.
     */
    ReplicateConfiguration&
    AddCrossClusterTopology(CrossClusterTopology&& topology);

 private:
    std::vector<MilvusCluster> clusters_;
    std::vector<CrossClusterTopology> cross_cluster_topologies_;
};

}  // namespace milvus
