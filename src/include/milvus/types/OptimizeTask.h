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

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../Status.h"
#include "../response/utility/OptimizeResponse.h"

namespace milvus {

class MilvusClientV2Impl;

/**
 * @brief Task object returned by MilvusClientV2::Optimize().
 */
class OptimizeTask : public std::enable_shared_from_this<OptimizeTask> {
 public:
    /**
     * @brief Constructor
     */
    OptimizeTask();

    /**
     * @brief Destructor
     */
    ~OptimizeTask();

    /**
     * @brief Wait for task result. Timeout zero means wait forever.
     */
    Status
    GetResult(OptimizeResponse& response, int64_t timeout_ms = 0);

    /**
     * @brief Cancel the task cooperatively.
     */
    bool
    Cancel();

    /**
     * @brief Whether the task is done.
     */
    bool
    IsDone() const;

    /**
     * @brief Whether the task is cancelled.
     */
    bool
    IsCancelled() const;

    /**
     * @brief Current progress message.
     */
    std::string
    CurrentProgress() const;

    /**
     * @brief Progress message history.
     */
    std::vector<std::string>
    ProgressHistory() const;

    /**
     * @brief Final task status if done, otherwise OK.
     */
    Status
    TaskStatus() const;

 private:
    friend class MilvusClientV2Impl;

    using Worker = std::function<Status(OptimizeResponse&)>;

    Status
    Start(Worker worker);

    bool
    ShouldCancel() const;

    void
    AddProgress(const std::string& progress);

    void
    Complete(const Status& status, OptimizeResponse&& response);

    Status
    CancelledStatus() const;

 private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool done_{false};
    bool cancelled_{false};
    Status status_;
    OptimizeResponse response_;
    std::vector<std::string> progress_history_;
    std::thread worker_;
};

using OptimizeTaskPtr = std::shared_ptr<OptimizeTask>;

}  // namespace milvus
