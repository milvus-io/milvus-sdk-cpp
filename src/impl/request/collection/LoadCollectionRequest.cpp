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

#include "milvus/request/collection/LoadCollectionRequest.h"

namespace milvus {

bool
LoadCollectionRequest::Sync() const {
    return sync_;
}

void
LoadCollectionRequest::SetSync(bool sync) {
    sync_ = sync;
}

LoadCollectionRequest&
LoadCollectionRequest::WithSync(bool sync) {
    SetSync(sync);
    return *this;
}

int64_t
LoadCollectionRequest::ReplicaNum() const {
    return replica_num_;
}

void
LoadCollectionRequest::SetReplicaNum(int64_t replica_num) {
    replica_num_ = replica_num;
}

LoadCollectionRequest&
LoadCollectionRequest::WithReplicaNum(int64_t replica_num) {
    SetReplicaNum(replica_num);
    return *this;
}

int64_t
LoadCollectionRequest::TimeoutMs() const {
    return timeout_ms_;
}

void
LoadCollectionRequest::SetTimeoutMs(int64_t timeout_ms) {
    timeout_ms_ = timeout_ms;
}

LoadCollectionRequest&
LoadCollectionRequest::WithTimeoutMs(int64_t timeout_ms) {
    SetTimeoutMs(timeout_ms);
    return *this;
}

bool
LoadCollectionRequest::Refresh() const {
    return refresh_;
}

void
LoadCollectionRequest::SetRefresh(bool refresh) {
    refresh_ = refresh;
}

LoadCollectionRequest&
LoadCollectionRequest::WithRefresh(bool refresh) {
    SetRefresh(refresh);
    return *this;
}

const std::set<std::string>&
LoadCollectionRequest::LoadFields() const {
    return load_feilds_;
}

void
LoadCollectionRequest::SetLoadFields(const std::set<std::string>& load_fields) {
    load_feilds_ = load_fields;
}

LoadCollectionRequest&
LoadCollectionRequest::WithLoadFields(const std::set<std::string>& load_fields) {
    SetLoadFields(load_fields);
    return *this;
}

LoadCollectionRequest&
LoadCollectionRequest::AddLoadField(const std::string& field_name) {
    load_feilds_.insert(field_name);
    return *this;
}

bool
LoadCollectionRequest::SkipDynamicField() const {
    return skip_dynamic_field_;
}

void
LoadCollectionRequest::SetSkipDynamicField(bool skip_dynamic_field) {
    skip_dynamic_field_ = skip_dynamic_field;
}

LoadCollectionRequest&
LoadCollectionRequest::WithSkipDynamicField(bool skip_dynamic_field) {
    SetSkipDynamicField(skip_dynamic_field);
    return *this;
}

const std::set<std::string>&
LoadCollectionRequest::TargetResourceGroups() const {
    return target_resource_groups_;
}

void
LoadCollectionRequest::SetTargetResourceGroups(const std::set<std::string>& target_resource_groups) {
    target_resource_groups_ = target_resource_groups;
}

LoadCollectionRequest&
LoadCollectionRequest::WithTargetResourceGroups(const std::set<std::string>& target_resource_groups) {
    SetTargetResourceGroups(target_resource_groups);
    return *this;
}

}  // namespace milvus
