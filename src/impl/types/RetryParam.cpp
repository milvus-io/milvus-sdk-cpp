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

#include "milvus/types/RetryParam.h"

namespace milvus {

RetryParam&
RetryParam::operator=(const RetryParam& other) {
    if (this != &other) {
        max_retry_times_ = other.max_retry_times_;
        max_retry_timeout_ms_ = other.max_retry_timeout_ms_;
        initial_backoff_ms_ = other.initial_backoff_ms_;
        max_backoff_ms_ = other.max_backoff_ms_;
        backoff_multiplier_ = other.backoff_multiplier_;
        retry_on_ratelimit_ = other.retry_on_ratelimit_;
    }
    return *this;
}

uint64_t
RetryParam::MaxRetryTimes() const {
    return max_retry_times_;
}

void
RetryParam::SetMaxRetryTimes(uint64_t max_retry_times) {
    max_retry_times_ = max_retry_times;
}

RetryParam&
RetryParam::WithMaxRetryTimes(uint64_t max_retry_times) {
    SetMaxRetryTimes(max_retry_times);
    return *this;
}

uint64_t
RetryParam::MaxRetryTimeoutMs() const {
    return max_retry_timeout_ms_;
}

void
RetryParam::SetMaxRetryTimeoutMs(uint64_t max_retry_timeout_ms) {
    max_retry_timeout_ms_ = max_retry_timeout_ms;
}

RetryParam&
RetryParam::WithMaxRetryTimeoutMs(uint64_t max_retry_timeout_ms) {
    SetMaxRetryTimeoutMs(max_retry_timeout_ms);
    return *this;
}

uint64_t
RetryParam::InitialBackOffMs() const {
    return initial_backoff_ms_;
}

void
RetryParam::SetInitialBackOffMs(uint64_t initial_backoff_ms) {
    if (initial_backoff_ms > 0) {
        initial_backoff_ms_ = initial_backoff_ms;
    }
}

RetryParam&
RetryParam::WithInitialBackOffMs(uint64_t initial_backoff_ms) {
    SetInitialBackOffMs(initial_backoff_ms);
    return *this;
}

uint64_t
RetryParam::MaxBackOffMs() const {
    return max_backoff_ms_;
}

void
RetryParam::SetMaxBackOffMs(uint64_t max_backoff_ms) {
    if (max_backoff_ms > 0) {
        max_backoff_ms_ = max_backoff_ms;
    }
}

RetryParam&
RetryParam::WithMaxBackOffMs(uint64_t max_backoff_ms) {
    SetMaxBackOffMs(max_backoff_ms);
    return *this;
}

uint64_t
RetryParam::BackOffMultiplier() const {
    return backoff_multiplier_;
}

void
RetryParam::SetBackOffMultiplier(uint64_t backoff_multiplier) {
    if (backoff_multiplier > 0) {
        backoff_multiplier_ = backoff_multiplier;
    }
}

RetryParam&
RetryParam::WithBackOffMultiplier(uint64_t backoff_multiplier) {
    SetBackOffMultiplier(backoff_multiplier);
    return *this;
}

bool
RetryParam::RetryOnRateLimit() const {
    return retry_on_ratelimit_;
}

void
RetryParam::SetRetryOnRateLimit(bool retry_on_ratelimit) {
    retry_on_ratelimit_ = retry_on_ratelimit;
}

RetryParam&
RetryParam::WithRetryOnRateLimit(bool retry_on_ratelimit) {
    SetRetryOnRateLimit(retry_on_ratelimit);
    return *this;
}

}  // namespace milvus
