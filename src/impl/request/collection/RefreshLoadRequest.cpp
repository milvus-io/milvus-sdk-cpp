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

#include "milvus/request/collection/RefreshLoadRequest.h"

namespace milvus {

bool
RefreshLoadRequest::Sync() const {
    return sync_;
}

void
RefreshLoadRequest::SetSync(bool sync) {
    sync_ = sync;
}

RefreshLoadRequest&
RefreshLoadRequest::WithSync(bool sync) {
    SetSync(sync);
    return *this;
}

int64_t
RefreshLoadRequest::TimeoutMs() const {
    return timeout_ms_;
}

void
RefreshLoadRequest::SetTimeoutMs(int64_t timeout_ms) {
    timeout_ms_ = timeout_ms;
}

RefreshLoadRequest&
RefreshLoadRequest::WithTimeoutMs(int64_t timeout_ms) {
    SetTimeoutMs(timeout_ms);
    return *this;
}

}  // namespace milvus
