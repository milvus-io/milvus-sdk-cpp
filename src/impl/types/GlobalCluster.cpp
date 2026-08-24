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

#include "GlobalCluster.h"

namespace milvus {

ClusterInfo::ClusterInfo(std::string cluster_id, std::string endpoint, int capability)
    : cluster_id_(std::move(cluster_id)), endpoint_(std::move(endpoint)), capability_(capability) {
}

const std::string&
ClusterInfo::ClusterId() const {
    return cluster_id_;
}

const std::string&
ClusterInfo::Endpoint() const {
    return endpoint_;
}

int
ClusterInfo::Capability() const {
    return capability_;
}

bool
ClusterInfo::IsPrimary() const {
    return (capability_ & ClusterCapability::WRITABLE) != 0;
}

GlobalTopology::GlobalTopology(int64_t version, std::vector<ClusterInfo> clusters)
    : version_(version), clusters_(std::move(clusters)) {
}

int64_t
GlobalTopology::Version() const {
    return version_;
}

const std::vector<ClusterInfo>&
GlobalTopology::Clusters() const {
    return clusters_;
}

const ClusterInfo*
GlobalTopology::Primary() const {
    for (const auto& cluster : clusters_) {
        if (cluster.IsPrimary()) {
            return &cluster;
        }
    }
    return nullptr;
}

}  // namespace milvus
