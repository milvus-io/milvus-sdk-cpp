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

#include "milvus/request/snapshot/RestoreSnapshotRequest.h"

namespace milvus {

const std::string&
RestoreSnapshotRequest::SnapshotName() const {
    return snapshot_name_;
}

void
RestoreSnapshotRequest::SetSnapshotName(const std::string& snapshot_name) {
    snapshot_name_ = snapshot_name;
}

RestoreSnapshotRequest&
RestoreSnapshotRequest::WithSnapshotName(const std::string& snapshot_name) {
    SetSnapshotName(snapshot_name);
    return *this;
}

const std::string&
RestoreSnapshotRequest::SourceDatabaseName() const {
    return source_db_name_;
}

void
RestoreSnapshotRequest::SetSourceDatabaseName(const std::string& db_name) {
    source_db_name_ = db_name;
}

RestoreSnapshotRequest&
RestoreSnapshotRequest::WithSourceDatabaseName(const std::string& db_name) {
    SetSourceDatabaseName(db_name);
    return *this;
}

const std::string&
RestoreSnapshotRequest::SourceCollectionName() const {
    return source_collection_name_;
}

void
RestoreSnapshotRequest::SetSourceCollectionName(const std::string& collection_name) {
    source_collection_name_ = collection_name;
}

RestoreSnapshotRequest&
RestoreSnapshotRequest::WithSourceCollectionName(const std::string& collection_name) {
    SetSourceCollectionName(collection_name);
    return *this;
}

const std::string&
RestoreSnapshotRequest::TargetDatabaseName() const {
    return target_db_name_;
}

void
RestoreSnapshotRequest::SetTargetDatabaseName(const std::string& db_name) {
    target_db_name_ = db_name;
}

RestoreSnapshotRequest&
RestoreSnapshotRequest::WithTargetDatabaseName(const std::string& db_name) {
    SetTargetDatabaseName(db_name);
    return *this;
}

const std::string&
RestoreSnapshotRequest::TargetCollectionName() const {
    return target_collection_name_;
}

void
RestoreSnapshotRequest::SetTargetCollectionName(const std::string& collection_name) {
    target_collection_name_ = collection_name;
}

RestoreSnapshotRequest&
RestoreSnapshotRequest::WithTargetCollectionName(const std::string& collection_name) {
    SetTargetCollectionName(collection_name);
    return *this;
}

}  // namespace milvus
