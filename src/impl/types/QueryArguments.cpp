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

#include "milvus/types/QueryArguments.h"

namespace milvus {

const std::string&
QueryArguments::CollectionName() const {
    return collection_name_;
}

Status
QueryArguments::SetCollectionName(std::string collection_name) {
    if (collection_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Collection name cannot be empty!"};
    }
    collection_name_ = std::move(collection_name);
    return Status::OK();
}

const std::set<std::string>&
QueryArguments::PartitionNames() const {
    return partition_names_;
}

Status
QueryArguments::AddPartitionName(std::string partition_name) {
    if (partition_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Partition name cannot be empty!"};
    }

    partition_names_.emplace(std::move(partition_name));
    return Status::OK();
}

const std::set<std::string>&
QueryArguments::OutputFields() const {
    return output_field_names_;
}

Status
QueryArguments::AddOutputField(std::string field_name) {
    if (field_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
    }

    output_field_names_.emplace(std::move(field_name));
    return Status::OK();
}

const std::string&
QueryArguments::Expression() const {
    return filter_expression_;
}

Status
QueryArguments::SetExpression(std::string expression) {
    if (expression.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Filter expression cannot be empty!"};
    }

    filter_expression_ = std::move(expression);
    return Status::OK();
}

uint64_t
QueryArguments::TravelTimestamp() const {
    return travel_timestamp_;
}

Status
QueryArguments::SetTravelTimestamp(uint64_t timestamp) {
    travel_timestamp_ = timestamp;
    return Status::OK();
}

uint64_t
QueryArguments::GuaranteeTimestamp() const {
    return guarantee_timestamp_;
}

Status
QueryArguments::SetGuaranteeTimestamp(uint64_t timestamp) {
    guarantee_timestamp_ = timestamp;
    return Status::OK();
}

}  // namespace milvus
