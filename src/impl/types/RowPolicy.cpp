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

#include "milvus/types/RowPolicy.h"

namespace milvus {

const std::string&
RowPolicy::PolicyName() const {
    return policy_name_;
}

void
RowPolicy::SetPolicyName(const std::string& name) {
    policy_name_ = name;
}

RowPolicyType
RowPolicy::PolicyType() const {
    return policy_type_;
}

void
RowPolicy::SetPolicyType(RowPolicyType policy_type) {
    policy_type_ = policy_type;
}

const std::vector<RowPolicyAction>&
RowPolicy::Actions() const {
    return actions_;
}

void
RowPolicy::SetActions(std::vector<RowPolicyAction>&& actions) {
    actions_ = std::move(actions);
}

const std::string&
RowPolicy::UsingExpr() const {
    return using_expr_;
}

void
RowPolicy::SetUsingExpr(const std::string& expr) {
    using_expr_ = expr;
}

const std::string&
RowPolicy::CheckExpr() const {
    return check_expr_;
}

void
RowPolicy::SetCheckExpr(const std::string& expr) {
    check_expr_ = expr;
}

const std::string&
RowPolicy::Description() const {
    return description_;
}

void
RowPolicy::SetDescription(const std::string& description) {
    description_ = description;
}

int64_t
RowPolicy::CreatedAt() const {
    return created_at_;
}

void
RowPolicy::SetCreatedAt(int64_t created_at) {
    created_at_ = created_at;
}

}  // namespace milvus
