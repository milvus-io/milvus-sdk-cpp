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

#include "milvus/request/dql/DQLRequestBase.h"

#include <memory>

namespace milvus {

const std::string&
DQLRequestBase::DatabaseName() const {
    return db_name_;
}

void
DQLRequestBase::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

DQLRequestBase&
DQLRequestBase::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

const std::string&
DQLRequestBase::CollectionName() const {
    return collection_name_;
}

void
DQLRequestBase::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

const std::set<std::string>&
DQLRequestBase::PartitionNames() const {
    return partition_names_;
}

void
DQLRequestBase::SetPartitionNames(std::set<std::string>&& partition_names) {
    partition_names_ = std::move(partition_names);
}

const std::set<std::string>&
DQLRequestBase::OutputFieldNames() const {
    return output_field_names_;
}

void
DQLRequestBase::SetOutputFieldNames(std::set<std::string>&& output_field_names) {
    output_field_names_ = std::move(output_field_names);
}

ConsistencyLevel
DQLRequestBase::ConsistencyLevelValue() const {
    return consistency_level_;
}

void
DQLRequestBase::SetConsistencyLevel(ConsistencyLevel consistency_level) {
    consistency_level_ = consistency_level;
}

}  // namespace milvus
