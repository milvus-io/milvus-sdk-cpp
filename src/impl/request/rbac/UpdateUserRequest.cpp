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

#include "milvus/request/rbac/UpdateUserRequest.h"

namespace milvus {

const std::string&
UpdateUserRequest::UserName() const {
    return user_name_;
}

void
UpdateUserRequest::SetUserName(const std::string& user_name) {
    user_name_ = user_name;
}

UpdateUserRequest&
UpdateUserRequest::WithUserName(const std::string& user_name) {
    SetUserName(user_name);
    return *this;
}

const std::string&
UpdateUserRequest::Description() const {
    return description_;
}

void
UpdateUserRequest::SetDescription(const std::string& description) {
    description_ = description;
}

UpdateUserRequest&
UpdateUserRequest::WithDescription(const std::string& description) {
    SetDescription(description);
    return *this;
}

}  // namespace milvus
