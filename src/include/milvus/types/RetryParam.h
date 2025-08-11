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

namespace milvus {

/**
 * @brief Parameters for RPC interfaces retry machinery.
 * For some special server error codes, such as RateLimit, SDK will continue retry RPC call
 * until the server return response. For network error or unrecoverable error, SDK will slip
 * retry and return errors.
 */
class RetryParam {
 public:
    RetryParam() = default;

    RetryParam&
    operator=(const RetryParam&);

    /**
     * @brief Get max retry times.
     */
    uint64_t
    MaxRetryTimes() const;

    /**
     * @brief Set max retry times.
     */
    void
    SetMaxRetryTimes(uint64_t max_retry_times);

    /**
     * @brief Set max retry times.
     */
    RetryParam&
    WithMaxRetryTimes(uint64_t max_retry_times);

    /**
     * @brief Get maximum retry timeout in milliseconds.
     */
    uint64_t
    MaxRetryTimeoutMs() const;

    /**
     * @brief Set maximum retry timeout in milliseconds.
     */
    void
    SetMaxRetryTimeoutMs(uint64_t max_retry_timeout_ms);

    /**
     * @brief Set maximum retry timeout in milliseconds.
     */
    RetryParam&
    WithMaxRetryTimeoutMs(uint64_t max_retry_timeout_ms);

    /**
     * @brief Get initial backOff in milliseconds.
     */
    uint64_t
    InitialBackOffMs() const;

    /**
     * @brief Set initial backOff in milliseconds.
     * @param initial_backoff_ms the initial time interval between retry calls, must be greater than 0
     */
    void
    SetInitialBackOffMs(uint64_t initial_backoff_ms);

    /**
     * @brief Set initial backOff in milliseconds.
     * @param initial_backoff_ms the initial time interval between retry calls, must be greater than 0
     */
    RetryParam&
    WithInitialBackOffMs(uint64_t initial_backoff_ms);

    /**
     * @brief Get maximum backOff in milliseconds.
     */
    uint64_t
    MaxBackOffMs() const;

    /**
     * @brief Set maximum backOff in milliseconds.
     * @param max_backoff_ms the maximum time interval between retry calls, must be greater than 0
     */
    void
    SetMaxBackOffMs(uint64_t max_backoff_ms);

    /**
     * @brief Set maximum backOff in milliseconds.
     * @param max_backoff_ms the maximum time interval between retry calls, must be greater than 0
     */
    RetryParam&
    WithMaxBackOffMs(uint64_t max_backoff_ms);

    /**
     * @brief Get backOff multiplier.
     */
    uint64_t
    BackOffMultiplier() const;

    /**
     * @brief Set backOff multiplier, automatically increase the time interval between retry calls.
     * @param backoff_multiplier the multiplier for time interval between retry calls, must be greater than 0
     */
    void
    SetBackOffMultiplier(uint64_t backoff_multiplier);

    /**
     * @brief Set backOff multiplier, automatically increase the time interval between retry calls.
     * @param backoff_multiplier the multiplier for time interval between retry calls, must be greater than 0
     */
    RetryParam&
    WithBackOffMultiplier(uint64_t backoff_multiplier);

    /**
     * @brief Get retry for ratelimit or not.
     */
    bool
    RetryOnRateLimit() const;

    /**
     * @brief Set retry for ratelimit.
     */
    void
    SetRetryOnRateLimit(bool retry_on_ratelimit);

    /**
     * @brief Set retry for ratelimit.
     */
    RetryParam&
    WithRetryOnRateLimit(bool retry_on_ratelimit);

 private:
    uint64_t max_retry_times_ = 75;
    uint64_t max_retry_timeout_ms_ = 0;  // uints: millisecond
    uint64_t initial_backoff_ms_ = 10;   // uints: millisecond
    uint64_t max_backoff_ms_ = 3000;     // uints: millisecond
    uint64_t backoff_multiplier_ = 3;
    bool retry_on_ratelimit_ = true;
};

}  // namespace milvus
