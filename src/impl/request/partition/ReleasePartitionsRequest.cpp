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

#include "milvus/request/partition/ReleasePartitionsRequest.h"

namespace milvus {

const std::string&
ReleasePartitionsRequest::DatabaseName() const {
    return db_name_;
}

void
ReleasePartitionsRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

ReleasePartitionsRequest&
ReleasePartitionsRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::string&
ReleasePartitionsRequest::CollectionName() const {
    return collection_name_;
}

void
ReleasePartitionsRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

ReleasePartitionsRequest&
ReleasePartitionsRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

const std::set<std::string>&
ReleasePartitionsRequest::PartitionNames() const {
    return partition_names_;
}

void
ReleasePartitionsRequest::SetPartitionNames(const std::set<std::string>& partition_names) {
    partition_names_ = partition_names;
}

ReleasePartitionsRequest&
ReleasePartitionsRequest::WithPartitionNames(const std::set<std::string>& partition_names) {
    SetPartitionNames(partition_names);
    return *this;
}

ReleasePartitionsRequest&
ReleasePartitionsRequest::AddPartitionName(const std::string& partition_name) {
    partition_names_.insert(partition_name);
    return *this;
}

}  // namespace milvus
