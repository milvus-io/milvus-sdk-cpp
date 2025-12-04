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

#include "milvus/request/index/AlterIndexPropertiesRequest.h"

#include <memory>

namespace milvus {

AlterIndexPropertiesRequest&
AlterIndexPropertiesRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

AlterIndexPropertiesRequest&
AlterIndexPropertiesRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

const std::string&
AlterIndexPropertiesRequest::FieldName() const {
    return field_name_;
}

void
AlterIndexPropertiesRequest::SetFieldName(const std::string& field_name) {
    field_name_ = field_name;
}

AlterIndexPropertiesRequest&
AlterIndexPropertiesRequest::WithFieldName(const std::string& field_name) {
    SetFieldName(field_name);
    return *this;
}

const std::unordered_map<std::string, std::string>&
AlterIndexPropertiesRequest::Properties() const {
    return properties_;
}

void
AlterIndexPropertiesRequest::SetProperties(std::unordered_map<std::string, std::string>&& properties) {
    properties_ = std::move(properties);
}

AlterIndexPropertiesRequest&
AlterIndexPropertiesRequest::WithProperties(std::unordered_map<std::string, std::string>&& properties) {
    SetProperties(std::move(properties));
    return *this;
}

AlterIndexPropertiesRequest&
AlterIndexPropertiesRequest::AddProperty(const std::string& key, const std::string& property) {
    properties_.insert(std::make_pair(key, property));
    return *this;
}

}  // namespace milvus
