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

#include "milvus/response/utility/CancelRequestsResponse.h"

#include <utility>

namespace milvus {

const RunningRequestInfos&
CancelRequestsResponse::Canceled() const {
    return canceled_;
}

void
CancelRequestsResponse::SetCanceled(RunningRequestInfos&& canceled) {
    canceled_ = std::move(canceled);
}

const std::vector<std::string>&
CancelRequestsResponse::NotFound() const {
    return not_found_;
}

void
CancelRequestsResponse::SetNotFound(const std::vector<std::string>& not_found) {
    not_found_ = not_found;
}

void
CancelRequestsResponse::SetNotFound(std::vector<std::string>&& not_found) {
    not_found_ = std::move(not_found);
}

const std::vector<std::string>&
CancelRequestsResponse::Undetermined() const {
    return undetermined_;
}

void
CancelRequestsResponse::SetUndetermined(const std::vector<std::string>& undetermined) {
    undetermined_ = undetermined;
}

void
CancelRequestsResponse::SetUndetermined(std::vector<std::string>&& undetermined) {
    undetermined_ = std::move(undetermined);
}

const RunningRequestNodeResults&
CancelRequestsResponse::NodeResults() const {
    return node_results_;
}

void
CancelRequestsResponse::SetNodeResults(RunningRequestNodeResults&& node_results) {
    node_results_ = std::move(node_results);
}

}  // namespace milvus
