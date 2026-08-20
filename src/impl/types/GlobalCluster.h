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

namespace milvus {

/**
 * @brief Bit flags describing the capabilities of a Milvus cluster in a global-cluster topology.
 */
struct ClusterCapability {
    enum {
        READABLE = 0b01,                // cluster can serve read (query/search) traffic
        WRITABLE = 0b10,                // cluster can accept write (insert/upsert/DDL) traffic
        PRIMARY = READABLE | WRITABLE,  // read + write
    };
};

/**
 * @brief A single cluster within a global-cluster topology.
 */
class ClusterInfo {
 public:
    ClusterInfo(std::string cluster_id, std::string endpoint, int capability);

    /**
     * @brief Unique identifier of the cluster.
     */
    const std::string&
    ClusterId() const;

    /**
     * @brief Endpoint (host[:port]) of the cluster.
     */
    const std::string&
    Endpoint() const;

    /**
     * @brief Capability bit flags (see ClusterCapability).
     */
    int
    Capability() const;

    /**
     * @brief Whether this cluster is the primary (writable) cluster.
     */
    bool
    IsPrimary() const;

 private:
    std::string cluster_id_;
    std::string endpoint_;
    int capability_{0};
};

/**
 * @brief Global cluster topology returned by the /global-cluster/topology endpoint.
 */
class GlobalTopology {
 public:
    GlobalTopology() = default;

    GlobalTopology(int64_t version, std::vector<ClusterInfo> clusters);

    /**
     * @brief Version of the topology; a change triggers failover.
     */
    int64_t
    Version() const;

    /**
     * @brief All clusters in the topology.
     */
    const std::vector<ClusterInfo>&
    Clusters() const;

    /**
     * @brief Find the primary (writable) cluster.
     * @return pointer to the primary cluster, or nullptr if no writable cluster exists.
     */
    const ClusterInfo*
    Primary() const;

 private:
    int64_t version_{0};
    std::vector<ClusterInfo> clusters_;
};

}  // namespace milvus
