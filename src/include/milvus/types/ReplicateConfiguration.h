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

#include <string>
#include <vector>

#include "milvus/Export.h"

namespace milvus {

class MILVUS_SDK_API MilvusCluster {
 public:
    const std::string&
    ClusterID() const;

    void
    SetClusterID(const std::string& cluster_id);

    MilvusCluster&
    WithClusterID(const std::string& cluster_id);

    const std::string&
    URI() const;

    void
    SetURI(const std::string& uri);

    MilvusCluster&
    WithURI(const std::string& uri);

    const std::string&
    Token() const;

    void
    SetToken(const std::string& token);

    MilvusCluster&
    WithToken(const std::string& token);

    const std::vector<std::string>&
    PChannels() const;

    void
    SetPChannels(std::vector<std::string>&& pchannels);

    MilvusCluster&
    WithPChannels(std::vector<std::string>&& pchannels);

    MilvusCluster&
    AddPChannel(const std::string& pchannel);

 private:
    std::string cluster_id_;
    std::string uri_;
    std::string token_;
    std::vector<std::string> pchannels_;
};

class MILVUS_SDK_API CrossClusterTopology {
 public:
    const std::string&
    SourceClusterID() const;

    void
    SetSourceClusterID(const std::string& cluster_id);

    CrossClusterTopology&
    WithSourceClusterID(const std::string& cluster_id);

    const std::string&
    TargetClusterID() const;

    void
    SetTargetClusterID(const std::string& cluster_id);

    CrossClusterTopology&
    WithTargetClusterID(const std::string& cluster_id);

 private:
    std::string source_cluster_id_;
    std::string target_cluster_id_;
};

class MILVUS_SDK_API ReplicateConfiguration {
 public:
    const std::vector<MilvusCluster>&
    Clusters() const;

    void
    SetClusters(std::vector<MilvusCluster>&& clusters);

    ReplicateConfiguration&
    WithClusters(std::vector<MilvusCluster>&& clusters);

    ReplicateConfiguration&
    AddCluster(MilvusCluster&& cluster);

    const std::vector<CrossClusterTopology>&
    CrossClusterTopologies() const;

    void
    SetCrossClusterTopologies(std::vector<CrossClusterTopology>&& topologies);

    ReplicateConfiguration&
    WithCrossClusterTopologies(std::vector<CrossClusterTopology>&& topologies);

    ReplicateConfiguration&
    AddCrossClusterTopology(CrossClusterTopology&& topology);

 private:
    std::vector<MilvusCluster> clusters_;
    std::vector<CrossClusterTopology> cross_cluster_topologies_;
};

}  // namespace milvus
