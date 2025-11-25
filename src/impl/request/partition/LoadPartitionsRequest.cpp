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

#include "milvus/request/partition/LoadPartitionsRequest.h"

namespace milvus {

const std::string&
LoadPartitionsRequest::DatabaseName() const {
    return db_name_;
}

void
LoadPartitionsRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

LoadPartitionsRequest&
LoadPartitionsRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::string&
LoadPartitionsRequest::CollectionName() const {
    return collection_name_;
}

void
LoadPartitionsRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

LoadPartitionsRequest&
LoadPartitionsRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

const std::set<std::string>&
LoadPartitionsRequest::PartitionNames() const {
    return partition_names_;
}

void
LoadPartitionsRequest::SetPartitionNames(const std::set<std::string>& partition_names) {
    partition_names_ = partition_names;
}

LoadPartitionsRequest&
LoadPartitionsRequest::WithPartitionNames(const std::set<std::string>& partition_names) {
    SetPartitionNames(partition_names);
    return *this;
}

LoadPartitionsRequest&
LoadPartitionsRequest::AddPartitionName(const std::string& partition_name) {
    partition_names_.insert(partition_name);
    return *this;
}

bool
LoadPartitionsRequest::Sync() const {
    return sync_;
}

void
LoadPartitionsRequest::SetSync(bool sync) {
    sync_ = sync;
}

LoadPartitionsRequest&
LoadPartitionsRequest::WithSync(bool sync) {
    SetSync(sync);
    return *this;
}

int64_t
LoadPartitionsRequest::ReplicaNum() const {
    return replica_num_;
}

void
LoadPartitionsRequest::SetReplicaNum(int64_t replica_num) {
    replica_num_ = replica_num;
}

LoadPartitionsRequest&
LoadPartitionsRequest::WithReplicaNum(int64_t replica_num) {
    SetReplicaNum(replica_num);
    return *this;
}

int64_t
LoadPartitionsRequest::TimeoutMs() const {
    return timeout_ms_;
}

void
LoadPartitionsRequest::SetTimeoutMs(int64_t timeout_ms) {
    timeout_ms_ = timeout_ms;
}

LoadPartitionsRequest&
LoadPartitionsRequest::WithTimeoutMs(int64_t timeout_ms) {
    SetTimeoutMs(timeout_ms);
    return *this;
}

bool
LoadPartitionsRequest::Refresh() const {
    return refresh_;
}

void
LoadPartitionsRequest::SetRefresh(bool refresh) {
    refresh_ = refresh;
}

LoadPartitionsRequest&
LoadPartitionsRequest::WithRefresh(bool refresh) {
    SetRefresh(refresh);
    return *this;
}

const std::set<std::string>&
LoadPartitionsRequest::LoadFields() const {
    return load_feilds_;
}

void
LoadPartitionsRequest::SetLoadFields(const std::set<std::string>& load_fields) {
    load_feilds_ = load_fields;
}

LoadPartitionsRequest&
LoadPartitionsRequest::WithLoadFields(const std::set<std::string>& load_fields) {
    SetLoadFields(load_fields);
    return *this;
}

LoadPartitionsRequest&
LoadPartitionsRequest::AddLoadField(const std::string& load_field) {
    load_feilds_.insert(load_field);
    return *this;
}

bool
LoadPartitionsRequest::SkipDynamicField() const {
    return skip_dynamic_field_;
}

void
LoadPartitionsRequest::SetSkipDynamicField(bool skip_dynamic_field) {
    skip_dynamic_field_ = skip_dynamic_field;
}

LoadPartitionsRequest&
LoadPartitionsRequest::WithSkipDynamicField(bool skip_dynamic_field) {
    SetSkipDynamicField(skip_dynamic_field);
    return *this;
}

const std::set<std::string>&
LoadPartitionsRequest::TargetResourceGroups() const {
    return target_resource_groups_;
}

void
LoadPartitionsRequest::SetTargetResourceGroups(const std::set<std::string>& target_resource_groups) {
    target_resource_groups_ = target_resource_groups;
}

LoadPartitionsRequest&
LoadPartitionsRequest::WithTargetResourceGroups(const std::set<std::string>& target_resource_groups) {
    SetTargetResourceGroups(target_resource_groups);
    return *this;
}

LoadPartitionsRequest&
LoadPartitionsRequest::AddTargetResourceGroups(const std::string& target_resource_group) {
    target_resource_groups_.insert(target_resource_group);
    return *this;
}

}  // namespace milvus
