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

#include "milvus/types/ProgressMonitor.h"

namespace milvus {

Progress::Progress() = default;

Progress::Progress(uint32_t finished, uint32_t total) : finished_(finished), total_(total) {
}

bool
Progress::Done() const {
    return finished_ >= total_;
}

bool
operator==(const Progress& a, const Progress& b) {
    return a.finished_ == b.finished_ && a.total_ == b.total_;
}

ProgressMonitor::ProgressMonitor(uint32_t check_timeout) : check_timeout_(check_timeout) {
}

ProgressMonitor::ProgressMonitor() = default;

uint32_t
ProgressMonitor::CheckTimeout() const {
    return check_timeout_;
}

uint32_t
ProgressMonitor::CheckInterval() const {
    return check_interval_;
}

void
ProgressMonitor::SetCheckInterval(uint32_t check_interval) {
    check_interval_ = check_interval;
}

void
ProgressMonitor::DoProgress(Progress& p) const {
    if (callback_func_ != nullptr) {
        callback_func_(p);
    }
}

void
ProgressMonitor::SetCallbackFunc(CallbackFunc func) {
    callback_func_ = std::move(func);
}

ProgressMonitor
ProgressMonitor::NoWait() {
    return ProgressMonitor{0};
}

ProgressMonitor
ProgressMonitor::Forever() {
    return ProgressMonitor{std::numeric_limits<uint32_t>::max()};
}

}  // namespace milvus
