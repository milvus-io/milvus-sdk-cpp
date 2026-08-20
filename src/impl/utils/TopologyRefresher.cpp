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

#include "TopologyRefresher.h"

#include <chrono>
#include <thread>

#include "GlobalClusterUtils.h"

namespace milvus {

TopologyRefresher::TopologyRefresher(std::string endpoint, std::string token, int64_t initial_version,
                                     std::chrono::milliseconds interval, Callback on_change)
    : endpoint_(std::move(endpoint)),
      token_(std::move(token)),
      on_change_(std::move(on_change)),
      interval_(interval),
      current_version_(initial_version) {
}

TopologyRefresher::~TopologyRefresher() {
    Stop();
}

void
TopologyRefresher::Start() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (thread_.joinable()) {
        return;
    }
    stop_.store(false);
    thread_ = std::thread(&TopologyRefresher::refreshLoop, this);
}

void
TopologyRefresher::Stop() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_.store(true);
    }
    cv_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void
TopologyRefresher::TriggerRefresh() {
    refresh_now_.store(true);
    cv_.notify_one();
}

int64_t
TopologyRefresher::CurrentVersion() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return current_version_;
}

void
TopologyRefresher::refreshLoop() {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait_for(lock, interval_, [this]() { return stop_.load() || refresh_now_.load(); });
            if (stop_.load()) {
                break;
            }
        }
        refreshOnce();
        // coalesce: clear the trigger flag only after the fetch completes, so a TriggerRefresh()
        // arriving while the fetch was in flight does not queue an immediate re-fetch (a
        // sustained UNAVAILABLE burst must not turn into continuous topology polling)
        refresh_now_.store(false);
    }
}

void
TopologyRefresher::refreshOnce() {
    GlobalTopology topology;
    auto status = GlobalClusterUtils::FetchTopology(endpoint_, token_, topology, [this]() { return stop_.load(); });
    if (!status.IsOk()) {
        // keep the cached topology, retry next interval
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (topology.Version() == current_version_) {
            return;
        }
    }

    // run the change callback outside the lock; the new version is only committed after it
    // succeeds so a failed failover (e.g. the new primary is briefly UNAVAILABLE) is retried
    // on the next refresh instead of being skipped forever
    bool handled = true;
    if (on_change_) {
        handled = on_change_(topology);
    }
    if (handled) {
        std::lock_guard<std::mutex> lock(mtx_);
        current_version_ = topology.Version();
    }
}

}  // namespace milvus
