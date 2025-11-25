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

#include "milvus/request/utility/CompactRequest.h"

namespace milvus {

const std::string&
CompactRequest::DatabaseName() const {
    return db_name_;
}

void
CompactRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

CompactRequest&
CompactRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::string&
CompactRequest::CollectionName() const {
    return collection_name_;
}

void
CompactRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

CompactRequest&
CompactRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

bool
CompactRequest::ClusteringCompaction() const {
    return is_clustring_compaction_;
}

void
CompactRequest::SetClusteringCompaction(bool clustering_compaction) {
    is_clustring_compaction_ = clustering_compaction;
}

CompactRequest&
CompactRequest::WithClusteringCompaction(bool clustering_compaction) {
    SetClusteringCompaction(clustering_compaction);
    return *this;
}

}  // namespace milvus
