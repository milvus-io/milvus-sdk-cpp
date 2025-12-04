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

#include "milvus/request/dml/InsertRequest.h"

#include <memory>

namespace milvus {

const std::string&
InsertRequest::DatabaseName() const {
    return db_name_;
}

void
InsertRequest::SetDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
}

InsertRequest&
InsertRequest::WithDatabaseName(const std::string& db_name) {
    db_name_ = db_name;
    return *this;
}

const std::string&
InsertRequest::CollectionName() const {
    return collection_name_;
}

void
InsertRequest::SetCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
}

InsertRequest&
InsertRequest::WithCollectionName(const std::string& collection_name) {
    collection_name_ = collection_name;
    return *this;
}

const std::string&
InsertRequest::PartitionName() const {
    return partition_name_;
}

void
InsertRequest::SetPartitionName(const std::string& partition_name) {
    partition_name_ = partition_name;
}

InsertRequest&
InsertRequest::WithPartitionName(const std::string& partition_name) {
    partition_name_ = partition_name;
    return *this;
}

const std::vector<FieldDataPtr>&
InsertRequest::ColumnsData() const {
    return columns_data_;
}

void
InsertRequest::SetColumnsData(std::vector<FieldDataPtr>&& columns_data) {
    columns_data_ = std::move(columns_data);
}

InsertRequest&
InsertRequest::WithColumnsData(std::vector<FieldDataPtr>&& columns_data) {
    columns_data_ = std::move(columns_data);
    return *this;
}

InsertRequest&
InsertRequest::AddColumnData(const FieldDataPtr& column_data) {
    columns_data_.push_back(column_data);
    return *this;
}

const EntityRows&
InsertRequest::RowsData() const {
    return rows_data_;
}

void
InsertRequest::SetRowsData(EntityRows&& rows_data) {
    rows_data_ = std::move(rows_data);
}

InsertRequest&
InsertRequest::WithRowsData(EntityRows&& rows_data) {
    rows_data_ = std::move(rows_data);
    return *this;
}

InsertRequest&
InsertRequest::AddRowData(EntityRow&& row_data) {
    rows_data_.emplace_back(std::move(row_data));
    return *this;
}

}  // namespace milvus
