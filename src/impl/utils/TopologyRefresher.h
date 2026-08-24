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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "../types/GlobalCluster.h"

namespace milvus {

/**
 * @brief Background thread that periodically refreshes the global-cluster topology and
 * notifies a callback when the topology version changes.
 */
class TopologyRefresher {
 public:
    // returns true when the topology change was handled; the new version is only committed
    // (and the same version re-triggered on the next refresh) when the callback succeeds.
    // should_stop lets the callback abort a long-running action (e.g. the reconnect connect)
    // promptly when the refresher is being stopped.
    using Callback = std::function<bool(const GlobalTopology&, const std::function<bool()>& should_stop)>;

    /**
     * @param endpoint global-cluster endpoint used to fetch the topology
     * @param token authorization token
     * @param initial_version the topology version observed at connect time
     * @param interval refresh interval (default 5 minutes)
     * @param on_change invoked (from the refresher thread) when the topology version changes
     */
    TopologyRefresher(std::string endpoint, std::string token, int64_t initial_version,
                      std::chrono::milliseconds interval = std::chrono::minutes(5), Callback on_change = nullptr);

    ~TopologyRefresher();

    TopologyRefresher(const TopologyRefresher&) = delete;
    TopologyRefresher&
    operator=(const TopologyRefresher&) = delete;

    /**
     * @brief Start the background refresh loop (no-op if already running).
     */
    void
    Start();

    /**
     * @brief Stop the background refresh loop.
     */
    void
    Stop();

    /**
     * @brief Trigger an immediate asynchronous refresh (wakes the loop thread; coalesced).
     */
    void
    TriggerRefresh();

    /**
     * @brief The current cached topology version.
     */
    int64_t
    CurrentVersion() const;

 private:
    void
    refreshLoop();

    void
    refreshOnce();

    std::string endpoint_;
    std::string token_;
    Callback on_change_;
    std::chrono::milliseconds interval_;

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    int64_t current_version_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> refresh_now_{false};
    std::thread thread_;
};

}  // namespace milvus
