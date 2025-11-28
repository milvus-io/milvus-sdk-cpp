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

#include "milvus/request/alias/ListAliasesRequest.h"

namespace milvus {

// Constructor is default, so no implementation needed.

// Returns the database name
const std::string&
ListAliasesRequest::DatabaseName() const {
    return db_name_;
}

// Sets the database name
void
ListAliasesRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

// Fluent-style setter for the database name
ListAliasesRequest&
ListAliasesRequest::WithDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
    return *this;
}

// Returns the collection name
const std::string&
ListAliasesRequest::CollectionName() const {
    return collection_name_;
}

// Sets the collection name
void
ListAliasesRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

// Fluent-style setter for the collection name
ListAliasesRequest&
ListAliasesRequest::WithCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
    return *this;
}

}  // namespace milvus