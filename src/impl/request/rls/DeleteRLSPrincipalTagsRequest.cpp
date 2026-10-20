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

#include "milvus/request/rls/DeleteRLSPrincipalTagsRequest.h"

namespace milvus {

const std::string&
DeleteRLSPrincipalTagsRequest::DatabaseName() const {
    return db_name_;
}

void
DeleteRLSPrincipalTagsRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

DeleteRLSPrincipalTagsRequest&
DeleteRLSPrincipalTagsRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::string&
DeleteRLSPrincipalTagsRequest::CollectionName() const {
    return collection_name_;
}

void
DeleteRLSPrincipalTagsRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

DeleteRLSPrincipalTagsRequest&
DeleteRLSPrincipalTagsRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

const std::string&
DeleteRLSPrincipalTagsRequest::PrincipalName() const {
    return principal_name_;
}

void
DeleteRLSPrincipalTagsRequest::SetPrincipalName(const std::string& principal_name) {
    principal_name_ = principal_name;
}

DeleteRLSPrincipalTagsRequest&
DeleteRLSPrincipalTagsRequest::WithPrincipalName(const std::string& principal_name) {
    SetPrincipalName(principal_name);
    return *this;
}

const std::vector<std::string>&
DeleteRLSPrincipalTagsRequest::TagKeys() const {
    return tag_keys_;
}

void
DeleteRLSPrincipalTagsRequest::SetTagKeys(std::vector<std::string>&& tag_keys) {
    tag_keys_ = std::move(tag_keys);
}

DeleteRLSPrincipalTagsRequest&
DeleteRLSPrincipalTagsRequest::WithTagKeys(std::vector<std::string>&& tag_keys) {
    SetTagKeys(std::move(tag_keys));
    return *this;
}

}  // namespace milvus
