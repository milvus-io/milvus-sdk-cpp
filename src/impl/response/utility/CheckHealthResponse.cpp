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

#include "milvus/response/utility/CheckHealthResponse.h"

#include <memory>

namespace milvus {

bool
CheckHealthResponse::IsHealthy() const {
    return is_healthy_;
}

void
CheckHealthResponse::SetIsHealthy(bool healthy) {
    is_healthy_ = healthy;
}

const std::vector<std::string>&
CheckHealthResponse::Reasons() const {
    return reasons_;
}

void
CheckHealthResponse::SetReasons(std::vector<std::string>&& reasons) {
    reasons_ = std::move(reasons);
}

const std::vector<std::string>&
CheckHealthResponse::QuotaStates() const {
    return quota_states_;
}

void
CheckHealthResponse::SetQuotaStates(std::vector<std::string>&& states) {
    quota_states_ = std::move(states);
}

}  // namespace milvus
