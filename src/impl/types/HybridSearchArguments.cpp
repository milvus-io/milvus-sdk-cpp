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

#include "milvus/types/HybridSearchArguments.h"

#include <nlohmann/json.hpp>
#include <utility>

#include "../utils/Constants.h"

namespace milvus {

const std::string&
HybridSearchArguments::DatabaseName() const {
    return db_name_;
}

Status
HybridSearchArguments::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
    return Status::OK();
}

const std::string&
HybridSearchArguments::CollectionName() const {
    return collection_name_;
}

Status
HybridSearchArguments::SetCollectionName(std::string collection_name) {
    if (collection_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Collection name cannot be empty!"};
    }
    collection_name_ = std::move(collection_name);
    return Status::OK();
}

const std::set<std::string>&
HybridSearchArguments::PartitionNames() const {
    return partition_names_;
}

Status
HybridSearchArguments::AddPartitionName(std::string partition_name) {
    if (partition_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Partition name cannot be empty!"};
    }
    partition_names_.emplace(std::move(partition_name));
    return Status::OK();
}

const std::set<std::string>&
HybridSearchArguments::OutputFields() const {
    return output_field_names_;
}

Status
HybridSearchArguments::AddOutputField(std::string field_name) {
    if (field_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
    }

    output_field_names_.emplace(std::move(field_name));
    return Status::OK();
}

int64_t
HybridSearchArguments::Limit() const {
    return limit_;
}

Status
HybridSearchArguments::SetLimit(int64_t limit) {
    limit_ = limit;
    return Status::OK();
}

int64_t
HybridSearchArguments::Offset() const {
    return offset_;
}

Status
HybridSearchArguments::SetOffset(int64_t offset) {
    offset_ = offset;
    return Status::OK();
}

int
HybridSearchArguments::RoundDecimal() const {
    return round_decimal_;
}

Status
HybridSearchArguments::SetRoundDecimal(int round_decimal) {
    round_decimal_ = round_decimal;
    return Status::OK();
}

ConsistencyLevel
HybridSearchArguments::GetConsistencyLevel() const {
    return consistency_level_;
}

Status
HybridSearchArguments::SetConsistencyLevel(const ConsistencyLevel& level) {
    consistency_level_ = level;
    return Status::OK();
}

bool
HybridSearchArguments::IgnoreGrowing() const {
    return ignore_growing_;
}

Status
HybridSearchArguments::SetIgnoreGrowing(bool ignore_growing) {
    ignore_growing_ = ignore_growing;
    return Status::OK();
}

const std::vector<SubSearchRequestPtr>&
HybridSearchArguments::SubRequests() const {
    return sub_requests_;
}

Status
HybridSearchArguments::AddSubRequest(const SubSearchRequestPtr& request) {
    sub_requests_.emplace_back(request);
    return Status::OK();
}

FunctionPtr
HybridSearchArguments::Rerank() const {
    return function_;
}

Status
HybridSearchArguments::SetRerank(const FunctionPtr& rerank) {
    function_ = rerank;
    return Status::OK();
}

Status
HybridSearchArguments::Validate() const {
    for (auto& it : sub_requests_) {
        if (it == nullptr) {
            return {StatusCode::INVALID_AGUMENT, "Sub request can not be null!"};
        }
        auto status = (*it).Validate();
        if (!status.IsOk()) {
            return status;
        }
    }
    if (function_ == nullptr) {
        return {StatusCode::INVALID_AGUMENT, "Rerank function is undefined!"};
    }
    if (function_->GetFunctionType() != FunctionType::RERANK) {
        return {StatusCode::INVALID_AGUMENT, "Hybrid search only accepts RERANK function!"};
    }

    return Status::OK();
}

}  // namespace milvus
