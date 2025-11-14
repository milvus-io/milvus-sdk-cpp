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

#include "milvus/types/RunAnalyzerArguments.h"

namespace milvus {
const std::string&
RunAnalyzerArguments::DatabaseName() const {
    return db_name_;
}

Status
RunAnalyzerArguments::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
    return Status::OK();
}

RunAnalyzerArguments&
RunAnalyzerArguments::WithDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
    return *this;
}

const std::string&
RunAnalyzerArguments::CollectionName() const {
    return collection_name_;
}

Status
RunAnalyzerArguments::SetCollectionName(std::string collection_name) {
    collection_name_ = collection_name;
    return Status::OK();
}

RunAnalyzerArguments&
RunAnalyzerArguments::WithCollectionName(std::string collection_name) {
    collection_name_ = collection_name;
    return *this;
}

const std::string&
RunAnalyzerArguments::FieldName() const {
    return field_name_;
}

Status
RunAnalyzerArguments::SetFieldName(std::string field_name) {
    field_name_ = field_name;
    return Status::OK();
}

RunAnalyzerArguments&
RunAnalyzerArguments::WithFieldName(std::string field_name) {
    field_name_ = field_name;
    return *this;
}

const std::vector<std::string>&
RunAnalyzerArguments::Texts() const {
    return texts_;
}

Status
RunAnalyzerArguments::SetTexts(const std::vector<std::string>& texts) {
    texts_ = texts;
    return Status::OK();
}

RunAnalyzerArguments&
RunAnalyzerArguments::AddText(std::string text) {
    texts_.emplace_back(text);
    return *this;
}

const std::vector<std::string>&
RunAnalyzerArguments::AnalyzerNames() const {
    return analyzer_names_;
}

Status
RunAnalyzerArguments::SetAnalyzerNames(const std::vector<std::string>& names) {
    analyzer_names_ = names;
    return Status::OK();
}

RunAnalyzerArguments&
RunAnalyzerArguments::AddAnalyzerName(std::string name) {
    analyzer_names_.emplace_back(name);
    return *this;
}

const nlohmann::json&
RunAnalyzerArguments::AnalyzerParams() const {
    return analyzer_params_;
}

Status
RunAnalyzerArguments::SetAnalyzerParams(const nlohmann::json& params) {
    analyzer_params_ = params;
    return Status::OK();
}

RunAnalyzerArguments&
RunAnalyzerArguments::WithAnalyzerParams(const nlohmann::json& params) {
    analyzer_params_ = params;
    return *this;
}

bool
RunAnalyzerArguments::IsWithDetail() const {
    return with_detail_;
}

RunAnalyzerArguments&
RunAnalyzerArguments::WithDetail(bool with_detail) {
    with_detail_ = with_detail;
    return *this;
}

RunAnalyzerArguments&
RunAnalyzerArguments::WithHash(bool with_hash) {
    with_hash_ = with_hash;
    return *this;
}

bool
RunAnalyzerArguments::IsWithHash() const {
    return with_hash_;
}

}  // namespace milvus
