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

namespace milvus {
class TimeoutSetting {
 public:
    explicit TimeoutSetting(uint32_t waiting_timeout) : waiting_timeout_(waiting_timeout) {
    }

    TimeoutSetting() = default;

    uint32_t
    waiting_timeout() const {
        return waiting_timeout_;
    }

    uint32_t
    waiting_interval() const {
        return waiting_interval_;
    }

    void
    SetInterval(uint32_t waiting_interval) {
        waiting_interval_ = waiting_interval;
    }

 private:
    /**
     * @brief Waiting duration
     *
     * This value control the waiting interval. Unit: millisecond. Default value: 500 milliseconds.
     */
    uint32_t waiting_interval_ = 500;

    /**
     * @brief Sync load waiting duration
     *
     * This value control the waiting timeout. Unit: second. Default value: 60 seconds.
     */
    uint32_t waiting_timeout_ = 60;
};
}  // namespace milvus
