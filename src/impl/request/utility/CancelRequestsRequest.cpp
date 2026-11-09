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

#include "milvus/request/utility/CancelRequestsRequest.h"

#include <utility>

namespace milvus {

const std::vector<std::string>&
CancelRequestsRequest::RequestIds() const {
    return request_ids_;
}

void
CancelRequestsRequest::SetRequestIds(const std::vector<std::string>& request_ids) {
    request_ids_ = request_ids;
}

void
CancelRequestsRequest::SetRequestIds(std::vector<std::string>&& request_ids) {
    request_ids_ = std::move(request_ids);
}

CancelRequestsRequest&
CancelRequestsRequest::WithRequestIds(std::vector<std::string>&& request_ids) {
    request_ids_ = std::move(request_ids);
    return *this;
}

CancelRequestsRequest&
CancelRequestsRequest::AddRequestId(const std::string& request_id) {
    request_ids_.emplace_back(request_id);
    return *this;
}

const std::string&
CancelRequestsRequest::Reason() const {
    return reason_;
}

void
CancelRequestsRequest::SetReason(const std::string& reason) {
    reason_ = reason;
}

CancelRequestsRequest&
CancelRequestsRequest::WithReason(const std::string& reason) {
    reason_ = reason;
    return *this;
}

const std::string&
CancelRequestsRequest::User() const {
    return user_;
}

void
CancelRequestsRequest::SetUser(const std::string& user) {
    user_ = user;
}

CancelRequestsRequest&
CancelRequestsRequest::WithUser(const std::string& user) {
    user_ = user;
    return *this;
}

}  // namespace milvus
