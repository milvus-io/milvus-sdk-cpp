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

#include "milvus/request/resourcegroup/TransferReplicaRequest.h"

namespace milvus {

const std::string&
TransferReplicaRequest::DatabaseName() const {
    return db_name_;
}

void
TransferReplicaRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

TransferReplicaRequest&
TransferReplicaRequest::WithDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
    return *this;
}

const std::string&
TransferReplicaRequest::CollectionName() const {
    return collection_name_;
}

void
TransferReplicaRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

TransferReplicaRequest&
TransferReplicaRequest::WithCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
    return *this;
}

const std::string&
TransferReplicaRequest::SourceGroup() const {
    return source_group_;
}

void
TransferReplicaRequest::SetSourceGroup(const std::string& source_group) {
    source_group_ = source_group;
}

TransferReplicaRequest&
TransferReplicaRequest::WithSourceGroup(const std::string& source_group) {
    source_group_ = source_group;
    return *this;
}

const std::string&
TransferReplicaRequest::TargetGroup() const {
    return target_group_;
}

void
TransferReplicaRequest::SetTargetGroup(const std::string& target_group) {
    target_group_ = target_group;
}

TransferReplicaRequest&
TransferReplicaRequest::WithTargetGroup(const std::string& target_group) {
    target_group_ = target_group;
    return *this;
}

int64_t
TransferReplicaRequest::NumReplicas() const {
    return num_replicas_;
}

void
TransferReplicaRequest::SetNumReplicas(int64_t num) {
    num_replicas_ = num;
}

TransferReplicaRequest&
TransferReplicaRequest::WithNumReplicas(int64_t num) {
    num_replicas_ = num;
    return *this;
}

}  // namespace milvus
