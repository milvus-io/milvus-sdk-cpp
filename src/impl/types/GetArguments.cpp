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

#include "milvus/types/GetArguments.h"

namespace milvus {

const std::string&
GetArguments::CollectionName() const {
    return collection_name_;
}

Status
GetArguments::SetCollectionName(std::string collection_name) {
    if (collection_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Collection name cannot be empty!"};
    }
    collection_name_ = std::move(collection_name);
    return Status::OK();
}

const std::set<std::string>&
GetArguments::PartitionNames() const {
    return partition_names_;
}

Status
GetArguments::AddPartitionName(std::string partition_name) {
    if (partition_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Partition name cannot be empty!"};
    }

    partition_names_.emplace(std::move(partition_name));
    return Status::OK();
}

const std::set<std::string>&
GetArguments::OutputFields() const {
    return output_field_names_;
}

Status
GetArguments::AddOutputField(std::string field_name) {
    if (field_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
    }

    output_field_names_.emplace(std::move(field_name));
    return Status::OK();
}

const std::vector<int64_t>&
GetArguments::Ids() const {
    return ids_;
}

Status
GetArguments::SetIds(std::vector<int64_t> ids) {
    if (ids.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Ids cannot be empty!"};
    }
    ids_ = std::move(ids);
    return Status::OK();
}

}  // namespace milvus
