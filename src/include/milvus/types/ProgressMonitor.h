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

    bool
    Done() const {
        return finished_ >= total_;
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

    /**
     * @brief Set time duration to wait the progress complete.
     *
     * @param [in] check_timeout set the value to controls the time duration to wait the progress. Unit: second.
     */
    explicit ProgressMonitor(uint32_t check_timeout) : check_timeout_(check_timeout) {
    }

    /**
     * @brief Default progress setting. Default timeout value: 60 seconds.
     */
    ProgressMonitor() = default;

    uint32_t
    CheckTimeout() const {
        return check_timeout_;
    }

    uint32_t
    CheckInterval() const {
        return check_interval_;
    }

    /**
     * @brief Set time interval to check the progress state.
     *
     * @param [in] check_interval set value to controls the time interval to
     * check progress state. Unit: millisecond. Default value: 500 milliseconds.
     */
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

    /**
     * @brief Set call back function to receive progress notification.
     *
     * @param [in] func call back function to recieve progress notification.
     */
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
    uint32_t check_interval_{500};
    uint32_t check_timeout_{60};

    std::function<void(Progress&)> callback_func_;
};
}  // namespace milvus
