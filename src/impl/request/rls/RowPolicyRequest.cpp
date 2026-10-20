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

#include "milvus/request/rls/RowPolicyRequest.h"

namespace milvus {

const std::string&
RowPolicyRequest::DatabaseName() const {
    return db_name_;
}

void
RowPolicyRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

RowPolicyRequest&
RowPolicyRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::string&
RowPolicyRequest::CollectionName() const {
    return collection_name_;
}

void
RowPolicyRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

RowPolicyRequest&
RowPolicyRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

const std::string&
RowPolicyRequest::PolicyName() const {
    return policy_name_;
}

void
RowPolicyRequest::SetPolicyName(const std::string& policy_name) {
    policy_name_ = policy_name;
}

RowPolicyRequest&
RowPolicyRequest::WithPolicyName(const std::string& policy_name) {
    SetPolicyName(policy_name);
    return *this;
}

RowPolicyType
RowPolicyRequest::PolicyType() const {
    return policy_type_;
}

void
RowPolicyRequest::SetPolicyType(RowPolicyType policy_type) {
    policy_type_ = policy_type;
}

RowPolicyRequest&
RowPolicyRequest::WithPolicyType(RowPolicyType policy_type) {
    SetPolicyType(policy_type);
    return *this;
}

const std::vector<RowPolicyAction>&
RowPolicyRequest::Actions() const {
    return actions_;
}

void
RowPolicyRequest::SetActions(std::vector<RowPolicyAction>&& actions) {
    actions_ = std::move(actions);
}

RowPolicyRequest&
RowPolicyRequest::WithActions(std::vector<RowPolicyAction>&& actions) {
    SetActions(std::move(actions));
    return *this;
}

const std::string&
RowPolicyRequest::UsingExpr() const {
    return using_expr_;
}

void
RowPolicyRequest::SetUsingExpr(const std::string& expr) {
    using_expr_ = expr;
}

RowPolicyRequest&
RowPolicyRequest::WithUsingExpr(const std::string& expr) {
    SetUsingExpr(expr);
    return *this;
}

const std::string&
RowPolicyRequest::CheckExpr() const {
    return check_expr_;
}

void
RowPolicyRequest::SetCheckExpr(const std::string& expr) {
    check_expr_ = expr;
}

RowPolicyRequest&
RowPolicyRequest::WithCheckExpr(const std::string& expr) {
    SetCheckExpr(expr);
    return *this;
}

const std::string&
RowPolicyRequest::Description() const {
    return description_;
}

void
RowPolicyRequest::SetDescription(const std::string& description) {
    description_ = description;
}

RowPolicyRequest&
RowPolicyRequest::WithDescription(const std::string& description) {
    SetDescription(description);
    return *this;
}

}  // namespace milvus
