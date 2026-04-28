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

#include "milvus/request/collection/BatchDescribeCollectionsRequest.h"

#include <utility>

namespace milvus {

const std::string&
BatchDescribeCollectionsRequest::DatabaseName() const {
    return db_name_;
}

void
BatchDescribeCollectionsRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

BatchDescribeCollectionsRequest&
BatchDescribeCollectionsRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::vector<std::string>&
BatchDescribeCollectionsRequest::CollectionNames() const {
    return collection_names_;
}

void
BatchDescribeCollectionsRequest::SetCollectionNames(std::vector<std::string>&& collection_names) {
    collection_names_ = std::move(collection_names);
}

BatchDescribeCollectionsRequest&
BatchDescribeCollectionsRequest::WithCollectionNames(std::vector<std::string>&& collection_names) {
    SetCollectionNames(std::move(collection_names));
    return *this;
}

BatchDescribeCollectionsRequest&
BatchDescribeCollectionsRequest::AddCollectionName(const std::string& collection_name) {
    collection_names_.push_back(collection_name);
    return *this;
}

const std::vector<int64_t>&
BatchDescribeCollectionsRequest::CollectionIDs() const {
    return collection_ids_;
}

void
BatchDescribeCollectionsRequest::SetCollectionIDs(std::vector<int64_t>&& collection_ids) {
    collection_ids_ = std::move(collection_ids);
}

BatchDescribeCollectionsRequest&
BatchDescribeCollectionsRequest::WithCollectionIDs(std::vector<int64_t>&& collection_ids) {
    SetCollectionIDs(std::move(collection_ids));
    return *this;
}

BatchDescribeCollectionsRequest&
BatchDescribeCollectionsRequest::AddCollectionID(int64_t collection_id) {
    collection_ids_.push_back(collection_id);
    return *this;
}

}  // namespace milvus
