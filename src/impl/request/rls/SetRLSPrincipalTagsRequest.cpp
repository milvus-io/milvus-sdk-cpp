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

#include "milvus/request/rls/SetRLSPrincipalTagsRequest.h"

namespace milvus {

const std::string&
SetRLSPrincipalTagsRequest::DatabaseName() const {
    return db_name_;
}

void
SetRLSPrincipalTagsRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

SetRLSPrincipalTagsRequest&
SetRLSPrincipalTagsRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::string&
SetRLSPrincipalTagsRequest::CollectionName() const {
    return collection_name_;
}

void
SetRLSPrincipalTagsRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

SetRLSPrincipalTagsRequest&
SetRLSPrincipalTagsRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

const std::string&
SetRLSPrincipalTagsRequest::PrincipalName() const {
    return principal_name_;
}

void
SetRLSPrincipalTagsRequest::SetPrincipalName(const std::string& principal_name) {
    principal_name_ = principal_name;
}

SetRLSPrincipalTagsRequest&
SetRLSPrincipalTagsRequest::WithPrincipalName(const std::string& principal_name) {
    SetPrincipalName(principal_name);
    return *this;
}

const nlohmann::json&
SetRLSPrincipalTagsRequest::Tags() const {
    return tags_;
}

void
SetRLSPrincipalTagsRequest::SetTags(nlohmann::json tags) {
    tags_ = std::move(tags);
}

SetRLSPrincipalTagsRequest&
SetRLSPrincipalTagsRequest::WithTags(nlohmann::json tags) {
    SetTags(std::move(tags));
    return *this;
}

}  // namespace milvus
