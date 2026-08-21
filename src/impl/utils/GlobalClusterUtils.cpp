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
#include "GlobalClusterUtils.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <random>
#include <thread>

// CPPHTTPLIB_OPENSSL_SUPPORT is provided as a compile definition by the build (see src/CMakeLists.txt).
#include "Uri.h"
#include "cpp-httplib/httplib.h"
#include "milvus/thirdparty/nlohmann/json.hpp"

namespace milvus {

namespace {
constexpr const char* GLOBAL_CLUSTER_MARKER = "global-cluster";
constexpr const char* TOPOLOGY_PATH = "/global-cluster/topology";
constexpr int MAX_RETRIES = 3;
constexpr int64_t BASE_BACKOFF_MS = 1000;
constexpr int64_t MAX_BACKOFF_MS = 10000;
constexpr int REQUEST_TIMEOUT_SEC = 10;

// Normalize the endpoint into a full topology URL (default and force https, strip trailing slash).
std::string
NormalizeEndpoint(const std::string& endpoint) {
    std::string base = endpoint;
    // trim whitespace
    base.erase(base.begin(), std::find_if(base.begin(), base.end(), [](unsigned char c) { return !std::isspace(c); }));
    base.erase(std::find_if(base.rbegin(), base.rend(), [](unsigned char c) { return !std::isspace(c); }).base(),
               base.end());

    if (base.find("://") == std::string::npos) {
        // default to https (the topology API requires TLS in production, like pymilvus)
        base = "https://" + base;
    }
    while (!base.empty() && base.back() == '/') {
        base.pop_back();
    }
    return base;
}

int64_t
CalculateBackoff(int attempt) {
    int64_t backoff = BASE_BACKOFF_MS * (1L << (attempt - 1));  // 1s, 2s, 4s...
    backoff = std::min(backoff, MAX_BACKOFF_MS);
    // add ~10% jitter
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    auto jitter = static_cast<int64_t>(backoff * 0.1 * dist(rng));
    return backoff + jitter;
}
}  // namespace

bool
GlobalClusterUtils::IsGlobalEndpoint(const std::string& uri) {
    if (uri.empty()) {
        return false;
    }
    std::string lowered = uri;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered.find(GLOBAL_CLUSTER_MARKER) != std::string::npos;
}

std::string
GlobalClusterUtils::BuildTopologyUrl(const std::string& endpoint) {
    return NormalizeEndpoint(endpoint) + TOPOLOGY_PATH;
}

std::string
GlobalClusterUtils::BuildPrimaryUri(const std::string& global_uri, bool tls_enabled,
                                    const std::string& primary_endpoint) {
    if (primary_endpoint.find("://") != std::string::npos) {
        return primary_endpoint;
    }
    // default to https for scheme-less global URIs as well, matching BuildTopologyUrl()/NormalizeEndpoint()
    bool use_https =
        tls_enabled || global_uri.compare(0, 8, "https://") == 0 || global_uri.find("://") == std::string::npos;
    return use_https ? "https://" + primary_endpoint : "http://" + primary_endpoint;
}

Status
GlobalClusterUtils::ParseTopologyResponse(const std::string& body, GlobalTopology& topology) {
    try {
        auto root = nlohmann::json::parse(body);
        if (!root.is_object() || !root.contains("code") || !root["code"].is_number_integer()) {
            return {StatusCode::INVALID_ARGUMENT, "Invalid global topology response: missing 'code'"};
        }
        int code = root["code"].get<int>();
        if (code != 0) {
            std::string message = root.value("message", "unknown error");
            return {StatusCode::SERVER_FAILED,
                    "Global topology API returned error code " + std::to_string(code) + ": " + message};
        }
        if (!root.contains("data") || !root["data"].is_object()) {
            return {StatusCode::INVALID_ARGUMENT, "Invalid global topology response: missing 'data'"};
        }
        auto data = root["data"];
        // the server returns version either as a number or a numeric string (pymilvus parses it
        // with int(data["version"]) and its fixtures use "version": "1")
        if (!data.contains("version") || !data.contains("clusters") || !data["clusters"].is_array()) {
            return {StatusCode::INVALID_ARGUMENT, "Invalid global topology response: missing version or clusters"};
        }

        int64_t version = 0;
        auto& version_value = data["version"];
        if (version_value.is_number_integer()) {
            version = version_value.get<int64_t>();
        } else if (version_value.is_string()) {
            try {
                version = std::stoll(version_value.get<std::string>());
            } catch (const std::exception&) {
                return {StatusCode::INVALID_ARGUMENT, "Invalid global topology response: bad version"};
            }
        } else {
            return {StatusCode::INVALID_ARGUMENT, "Invalid global topology response: bad version"};
        }

        std::vector<ClusterInfo> clusters;
        for (const auto& cluster : data["clusters"]) {
            if (!cluster.is_object() || !cluster.contains("clusterId") || !cluster.contains("endpoint") ||
                !cluster.contains("capability")) {
                return {StatusCode::INVALID_ARGUMENT, "Invalid global topology response: malformed cluster entry"};
            }
            clusters.emplace_back(cluster["clusterId"].get<std::string>(), cluster["endpoint"].get<std::string>(),
                                  cluster["capability"].get<int>());
        }
        topology = GlobalTopology(version, std::move(clusters));
        return Status::OK();
    } catch (const nlohmann::json::exception& e) {
        return {StatusCode::INVALID_ARGUMENT, "Failed to parse global topology response: " + std::string(e.what())};
    }
}

Status
GlobalClusterUtils::FetchTopology(const std::string& endpoint, const std::string& token, GlobalTopology& topology,
                                  const std::function<bool()>& should_stop) {
    std::string url = BuildTopologyUrl(endpoint);
    URI uri;
    try {
        uri = ParseURI(url);
    } catch (const std::exception&) {
        return {StatusCode::INVALID_ARGUMENT, "Invalid global cluster endpoint: " + endpoint};
    }
    if (uri.host.empty()) {
        return {StatusCode::INVALID_ARGUMENT, "Invalid global cluster endpoint: " + endpoint};
    }
    // ParseURI defaults a non-https URI without an explicit port to the Milvus gRPC port (19530);
    // the topology REST API is plain HTTP(S), so use the scheme defaults instead.
    int port = uri.port;
    if (uri.scheme == "https") {
        if (port == 0) {
            port = 443;
        }
    } else if (port == 0 || port == 19530) {
        port = 80;
    }
    const std::string& scheme = uri.scheme;
    const std::string& host = uri.host;
    std::string path = uri.path.empty() ? "/" : uri.path;

    httplib::Headers headers;
    if (!token.empty()) {
        headers.emplace("Authorization", "Bearer " + token);
    }
    headers.emplace("Accept", "application/json");

    Status last_status{StatusCode::RPC_FAILED, "Failed to fetch global topology"};
    for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        if (should_stop && should_stop()) {
            return {StatusCode::RPC_FAILED, "Global topology fetch aborted"};
        }

        httplib::Result res;
        if (scheme == "https") {
            httplib::SSLClient client(host, port);
            client.set_connection_timeout(REQUEST_TIMEOUT_SEC, 0);
            client.set_read_timeout(REQUEST_TIMEOUT_SEC, 0);
            res = client.Get(path, headers);
        } else {
            httplib::Client client(host, port);
            client.set_connection_timeout(REQUEST_TIMEOUT_SEC, 0);
            client.set_read_timeout(REQUEST_TIMEOUT_SEC, 0);
            res = client.Get(path, headers);
        }

        if (res && res->status == 200) {
            auto parse_status = ParseTopologyResponse(res->body, topology);
            if (parse_status.IsOk()) {
                return parse_status;
            }
            if (parse_status.Code() == StatusCode::SERVER_FAILED) {
                // the API reported an explicit error (code != 0); do not retry
                return parse_status;
            }
            // malformed/truncated body: treat as a retryable transient failure (like pymilvus)
            last_status = parse_status;
        } else {
            std::string reason = res ? ("HTTP status: " + std::to_string(res->status)) : "connection failed";
            last_status =
                Status{StatusCode::RPC_FAILED, "Failed to fetch global topology (attempt " + std::to_string(attempt) +
                                                   "/" + std::to_string(MAX_RETRIES) + "): " + reason};
        }

        if (attempt < MAX_RETRIES) {
            // bail out promptly on shutdown rather than sleeping through the remaining backoff
            if (should_stop && should_stop()) {
                return {StatusCode::RPC_FAILED, "Global topology fetch aborted"};
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(CalculateBackoff(attempt)));
        }
    }
    return last_status;
}

}  // namespace milvus
