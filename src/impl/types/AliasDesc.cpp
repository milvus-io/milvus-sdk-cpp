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

#include "milvus/types/AliasDesc.h"

namespace milvus {

AliasDesc::AliasDesc() = default;

AliasDesc::AliasDesc(const std::string& db_name, const std::string& alias, const std::string& collection_name)
    : db_name_(db_name), alias_(alias), collection_name_(collection_name) {}

const std::string&
AliasDesc::GetDbName() const {
    return db_name_;
}

const std::string&
AliasDesc::GetAlias() const {
    return alias_;
}

const std::string&
AliasDesc::GetCollectionName() const {
    return collection_name_;
}

void
AliasDesc::SetDbName(const std::string& db_name) {
    db_name_ = db_name;
}

void
AliasDesc::SetAlias(const std::string& alias) {
    alias_ = alias;
}

void
AliasDesc::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

}  // namespace milvus
