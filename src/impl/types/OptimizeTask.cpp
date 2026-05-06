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

#include "milvus/types/OptimizeTask.h"

#include <chrono>
#include <exception>
#include <system_error>
#include <utility>

namespace milvus {

OptimizeTask::OptimizeTask() = default;

OptimizeTask::~OptimizeTask() = default;

Status
OptimizeTask::GetResult(OptimizeResponse& response, int64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (timeout_ms > 0) {
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return done_; })) {
            return {StatusCode::TIMEOUT, "Timeout waiting for optimization to complete"};
        }
    } else {
        cv_.wait(lock, [this] { return done_; });
    }

    response = response_;
    return status_;
}

bool
OptimizeTask::Cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cancelled_) {
        return true;
    }
    if (done_) {
        return false;
    }

    cancelled_ = true;
    progress_history_.push_back("cancelling");
    return true;
}

bool
OptimizeTask::IsDone() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_;
}

bool
OptimizeTask::IsCancelled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancelled_;
}

std::string
OptimizeTask::CurrentProgress() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (progress_history_.empty()) {
        return "";
    }
    return progress_history_.back();
}

std::vector<std::string>
OptimizeTask::ProgressHistory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return progress_history_;
}

Status
OptimizeTask::TaskStatus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

Status
OptimizeTask::Start(Worker worker) {
    auto self = shared_from_this();
    try {
        worker_ = std::thread([self, worker]() {
            OptimizeResponse response;
            try {
                auto status = worker(response);
                self->Complete(status, std::move(response));
            } catch (const std::exception& e) {
                auto status = Status{StatusCode::UNKNOWN_ERROR, "Optimization task failed: " + std::string(e.what())};
                self->Complete(status, std::move(response));
            } catch (...) {
                auto status = Status{StatusCode::UNKNOWN_ERROR, "Optimization task failed with unknown exception"};
                self->Complete(status, std::move(response));
            }
        });
        worker_.detach();
    } catch (const std::system_error& e) {
        OptimizeResponse response;
        auto status = Status{StatusCode::UNKNOWN_ERROR, "Failed to start optimization task: " + std::string(e.what())};
        Complete(status, std::move(response));
        return status;
    }
    return Status::OK();
}

bool
OptimizeTask::ShouldCancel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancelled_;
}

void
OptimizeTask::AddProgress(const std::string& progress) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!done_ && !cancelled_) {
        progress_history_.push_back(progress);
    }
}

void
OptimizeTask::Complete(const Status& status, OptimizeResponse&& response) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (done_) {
        return;
    }

    status_ = status;
    auto history = progress_history_;
    response.SetProgressHistory(std::move(history));
    if (response.StatusText().empty()) {
        response.SetStatusText(cancelled_ ? "cancelled" : (status.IsOk() ? "success" : "failed"));
    }
    response_ = std::move(response);
    done_ = true;
    cv_.notify_all();
}

Status
OptimizeTask::CancelledStatus() const {
    return {StatusCode::UNKNOWN_ERROR, "Optimization task was cancelled"};
}

}  // namespace milvus
