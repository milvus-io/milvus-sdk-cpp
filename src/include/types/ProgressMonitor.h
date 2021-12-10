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
#include <functional>
#include <limits>

namespace milvus {

struct Progress {
    Progress() = default;

    Progress(uint32_t finished, uint32_t total) : finished_(finished), total_(total) {
    }

    uint32_t finished_ = 0;
    uint32_t total_ = 0;
};

inline bool
operator==(const Progress& a, const Progress& b) {
    return a.finished_ == b.finished_ && a.total_ == b.total_;
}

class ProgressMonitor {
 public:
    using CallbackFunc = std::function<void(Progress&)>;

 public:
    explicit ProgressMonitor(uint32_t check_timeout) : check_timeout_(check_timeout) {
    }

    ProgressMonitor() = default;

    uint32_t
    CheckTimeout() const {
        return check_timeout_;
    }

    uint32_t
    CheckInterval() const {
        return check_interval_;
    }

    void
    SetCheckInterval(uint32_t check_interval) {
        check_interval_ = check_interval;
    }

    void
    DoProgress(Progress& p) const {
        if (callback_func_ != nullptr) {
            callback_func_(p);
        }
    }

    void
    SetCallbackFunc(const CallbackFunc& func) {
        callback_func_ = func;
    }

    static ProgressMonitor
    NoWait() {
        return ProgressMonitor{0};
    }

    static ProgressMonitor
    Forever() {
        return ProgressMonitor{std::numeric_limits<uint32_t>::max()};
    }

 private:
    /**
     * @brief Time interval to check the progress state
     *
     * This value controls the time interval to check progress state. Unit: millisecond. Default value: 500
     * milliseconds.
     */
    uint32_t check_interval_{500};

    /**
     * @brief Time duration to wait the progress complete
     *
     * This value controls the time duration to wait the progress. Unit: second. Default value: 60 seconds.
     */
    uint32_t check_timeout_{60};

    /**
     * @brief Time duration to wait the progress complete
     *
     * This value controls the time duration to wait the progress. Unit: second. Default value: 60 seconds.
     */
    std::function<void(Progress&)> callback_func_;
};
}  // namespace milvus
