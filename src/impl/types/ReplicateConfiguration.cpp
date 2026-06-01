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

#include "milvus/types/ReplicateConfiguration.h"

#include <utility>

namespace milvus {

const std::string&
MilvusCluster::ClusterID() const {
    return cluster_id_;
}

void
MilvusCluster::SetClusterID(const std::string& cluster_id) {
    cluster_id_ = cluster_id;
}

MilvusCluster&
MilvusCluster::WithClusterID(const std::string& cluster_id) {
    SetClusterID(cluster_id);
    return *this;
}

const std::string&
MilvusCluster::Uri() const {
    return uri_;
}

void
MilvusCluster::SetUri(const std::string& uri) {
    uri_ = uri;
}

MilvusCluster&
MilvusCluster::WithUri(const std::string& uri) {
    SetUri(uri);
    return *this;
}

const std::string&
MilvusCluster::Token() const {
    return token_;
}

void
MilvusCluster::SetToken(const std::string& token) {
    token_ = token;
}

MilvusCluster&
MilvusCluster::WithToken(const std::string& token) {
    SetToken(token);
    return *this;
}

const std::vector<std::string>&
MilvusCluster::PChannels() const {
    return pchannels_;
}

void
MilvusCluster::SetPChannels(std::vector<std::string>&& pchannels) {
    pchannels_ = std::move(pchannels);
}

MilvusCluster&
MilvusCluster::WithPChannels(std::vector<std::string>&& pchannels) {
    SetPChannels(std::move(pchannels));
    return *this;
}

MilvusCluster&
MilvusCluster::AddPChannel(const std::string& pchannel) {
    pchannels_.push_back(pchannel);
    return *this;
}

const std::string&
CrossClusterTopology::SourceClusterID() const {
    return source_cluster_id_;
}

void
CrossClusterTopology::SetSourceClusterID(const std::string& cluster_id) {
    source_cluster_id_ = cluster_id;
}

CrossClusterTopology&
CrossClusterTopology::WithSourceClusterID(const std::string& cluster_id) {
    SetSourceClusterID(cluster_id);
    return *this;
}

const std::string&
CrossClusterTopology::TargetClusterID() const {
    return target_cluster_id_;
}

void
CrossClusterTopology::SetTargetClusterID(const std::string& cluster_id) {
    target_cluster_id_ = cluster_id;
}

CrossClusterTopology&
CrossClusterTopology::WithTargetClusterID(const std::string& cluster_id) {
    SetTargetClusterID(cluster_id);
    return *this;
}

const std::string&
ReplicateMessageID::ID() const {
    return id_;
}

void
ReplicateMessageID::SetID(const std::string& id) {
    id_ = id;
}

ReplicateMessageID&
ReplicateMessageID::WithID(const std::string& id) {
    SetID(id);
    return *this;
}

const std::string&
ReplicateMessageID::WalName() const {
    return wal_name_;
}

void
ReplicateMessageID::SetWalName(const std::string& wal_name) {
    wal_name_ = wal_name;
}

ReplicateMessageID&
ReplicateMessageID::WithWalName(const std::string& wal_name) {
    SetWalName(wal_name);
    return *this;
}

const std::string&
ReplicateCheckpoint::ClusterID() const {
    return cluster_id_;
}

void
ReplicateCheckpoint::SetClusterID(const std::string& cluster_id) {
    cluster_id_ = cluster_id;
}

ReplicateCheckpoint&
ReplicateCheckpoint::WithClusterID(const std::string& cluster_id) {
    SetClusterID(cluster_id);
    return *this;
}

const std::string&
ReplicateCheckpoint::PChannel() const {
    return pchannel_;
}

void
ReplicateCheckpoint::SetPChannel(const std::string& pchannel) {
    pchannel_ = pchannel;
}

ReplicateCheckpoint&
ReplicateCheckpoint::WithPChannel(const std::string& pchannel) {
    SetPChannel(pchannel);
    return *this;
}

const ReplicateMessageID&
ReplicateCheckpoint::MessageID() const {
    return message_id_;
}

void
ReplicateCheckpoint::SetMessageID(ReplicateMessageID&& message_id) {
    message_id_ = std::move(message_id);
}

ReplicateCheckpoint&
ReplicateCheckpoint::WithMessageID(ReplicateMessageID&& message_id) {
    SetMessageID(std::move(message_id));
    return *this;
}

uint64_t
ReplicateCheckpoint::TimeTick() const {
    return time_tick_;
}

void
ReplicateCheckpoint::SetTimeTick(uint64_t time_tick) {
    time_tick_ = time_tick;
}

ReplicateCheckpoint&
ReplicateCheckpoint::WithTimeTick(uint64_t time_tick) {
    SetTimeTick(time_tick);
    return *this;
}

const std::vector<MilvusCluster>&
ReplicateConfiguration::Clusters() const {
    return clusters_;
}

void
ReplicateConfiguration::SetClusters(std::vector<MilvusCluster>&& clusters) {
    clusters_ = std::move(clusters);
}

ReplicateConfiguration&
ReplicateConfiguration::WithClusters(std::vector<MilvusCluster>&& clusters) {
    SetClusters(std::move(clusters));
    return *this;
}

ReplicateConfiguration&
ReplicateConfiguration::AddCluster(MilvusCluster&& cluster) {
    clusters_.emplace_back(std::move(cluster));
    return *this;
}

const std::vector<CrossClusterTopology>&
ReplicateConfiguration::CrossClusterTopologies() const {
    return cross_cluster_topologies_;
}

void
ReplicateConfiguration::SetCrossClusterTopologies(std::vector<CrossClusterTopology>&& topologies) {
    cross_cluster_topologies_ = std::move(topologies);
}

ReplicateConfiguration&
ReplicateConfiguration::WithCrossClusterTopologies(std::vector<CrossClusterTopology>&& topologies) {
    SetCrossClusterTopologies(std::move(topologies));
    return *this;
}

ReplicateConfiguration&
ReplicateConfiguration::AddCrossClusterTopology(CrossClusterTopology&& topology) {
    cross_cluster_topologies_.emplace_back(std::move(topology));
    return *this;
}

}  // namespace milvus
