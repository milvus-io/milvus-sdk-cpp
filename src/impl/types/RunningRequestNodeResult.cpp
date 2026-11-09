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

#include "milvus/types/RunningRequestNodeResult.h"

namespace milvus {

int64_t
RunningRequestNodeResult::NodeId() const {
    return node_id_;
}

void
RunningRequestNodeResult::SetNodeId(int64_t node_id) {
    node_id_ = node_id;
}

RunningRequestNodeResult&
RunningRequestNodeResult::WithNodeId(int64_t node_id) {
    node_id_ = node_id;
    return *this;
}

int64_t
RunningRequestNodeResult::ErrorCode() const {
    return error_code_;
}

void
RunningRequestNodeResult::SetErrorCode(int64_t error_code) {
    error_code_ = error_code;
}

RunningRequestNodeResult&
RunningRequestNodeResult::WithErrorCode(int64_t error_code) {
    error_code_ = error_code;
    return *this;
}

const std::string&
RunningRequestNodeResult::Reason() const {
    return reason_;
}

void
RunningRequestNodeResult::SetReason(const std::string& reason) {
    reason_ = reason;
}

RunningRequestNodeResult&
RunningRequestNodeResult::WithReason(const std::string& reason) {
    reason_ = reason;
    return *this;
}

bool
RunningRequestNodeResult::Unimplemented() const {
    return unimplemented_;
}

void
RunningRequestNodeResult::SetUnimplemented(bool unimplemented) {
    unimplemented_ = unimplemented;
}

RunningRequestNodeResult&
RunningRequestNodeResult::WithUnimplemented(bool unimplemented) {
    unimplemented_ = unimplemented;
    return *this;
}

bool
RunningRequestNodeResult::IsSuccess() const {
    return error_code_ == 0;
}

}  // namespace milvus
