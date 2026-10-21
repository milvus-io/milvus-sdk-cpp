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

#include "milvus/request/rls/GetRLSPrincipalTagsRequest.h"

namespace milvus {

const std::string&
GetRLSPrincipalTagsRequest::DatabaseName() const {
    return db_name_;
}

void
GetRLSPrincipalTagsRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

GetRLSPrincipalTagsRequest&
GetRLSPrincipalTagsRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::string&
GetRLSPrincipalTagsRequest::CollectionName() const {
    return collection_name_;
}

void
GetRLSPrincipalTagsRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

GetRLSPrincipalTagsRequest&
GetRLSPrincipalTagsRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

const std::string&
GetRLSPrincipalTagsRequest::PrincipalName() const {
    return principal_name_;
}

void
GetRLSPrincipalTagsRequest::SetPrincipalName(const std::string& principal_name) {
    principal_name_ = principal_name;
}

GetRLSPrincipalTagsRequest&
GetRLSPrincipalTagsRequest::WithPrincipalName(const std::string& principal_name) {
    SetPrincipalName(principal_name);
    return *this;
}

}  // namespace milvus
