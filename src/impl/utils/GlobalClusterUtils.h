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

#include <functional>
#include <string>

#include "../types/GlobalCluster.h"
#include "milvus/Status.h"

namespace milvus {

/**
 * @brief Utilities for the global-cluster topology endpoint.
 */
class GlobalClusterUtils {
 public:
    /**
     * @brief Whether a connection URI targets a global-cluster endpoint.
     */
    static bool
    IsGlobalEndpoint(const std::string& uri);

    /**
     * @brief Build the topology REST URL from a global endpoint.
     * e.g. "https://xxx.global-cluster.yyy.com:443" -> ".../global-cluster/topology"
     */
    static std::string
    BuildTopologyUrl(const std::string& endpoint);

    /**
     * @brief Build the connection URI for a primary cluster endpoint, preserving the scheme of
     * the original global endpoint (https when the global URI is https or TLS is enabled, http
     * otherwise). An explicit scheme in the primary endpoint is used as-is.
     */
    static std::string
    BuildPrimaryUri(const std::string& global_uri, bool tls_enabled, const std::string& primary_endpoint);

    /**
     * @brief Fetch and parse the global cluster topology via HTTP(S) GET.
     * @param should_stop optional predicate; when it returns true the fetch aborts promptly
     * (used to keep shutdown fast while a refresh is in flight).
     */
    static Status
    FetchTopology(const std::string& endpoint, const std::string& token, GlobalTopology& topology,
                  const std::function<bool()>& should_stop = {});

    /**
     * @brief Parse a topology response body into a GlobalTopology.
     * Exposed for testing.
     */
    static Status
    ParseTopologyResponse(const std::string& body, GlobalTopology& topology);
};

}  // namespace milvus
