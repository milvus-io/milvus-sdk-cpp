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

#include "milvus/request/dml/UpsertRequest.h"

#include <memory>

namespace milvus {

UpsertRequest&
UpsertRequest::WithDatabaseName(const std::string& db_name) {
    SetDatabaseName(db_name);
    return *this;
}

UpsertRequest&
UpsertRequest::WithCollectionName(const std::string& collection_name) {
    SetCollectionName(collection_name);
    return *this;
}

UpsertRequest&
UpsertRequest::WithPartitionName(const std::string& partition_name) {
    SetPartitionName(partition_name);
    return *this;
}

UpsertRequest&
UpsertRequest::WithColumnsData(std::vector<FieldDataPtr>&& columns_data) {
    SetColumnsData(std::move(columns_data));
    return *this;
}

UpsertRequest&
UpsertRequest::AddColumnData(const FieldDataPtr& column_data) {
    InsertRequest::AddColumnData(column_data);
    return *this;
}

UpsertRequest&
UpsertRequest::WithRowsData(EntityRows&& rows_data) {
    SetRowsData(std::move(rows_data));
    return *this;
}

UpsertRequest&
UpsertRequest::AddRowData(EntityRow&& row_data) {
    InsertRequest::AddRowData(std::move(row_data));
    return *this;
}

bool
UpsertRequest::PartialUpdate() const {
    for (const auto& field_op : field_ops_) {
        if (field_op.GetOpType() != FieldPartialUpdateOp::OpType::REPLACE) {
            return true;
        }
    }
    return partial_update_;
}

void
UpsertRequest::SetPartialUpdate(bool partial_update) {
    partial_update_ = partial_update;
}

UpsertRequest&
UpsertRequest::WithPartialUpdate(bool partial_update) {
    partial_update_ = partial_update;
    return *this;
}

const std::vector<FieldPartialUpdateOp>&
UpsertRequest::FieldOps() const {
    return field_ops_;
}

void
UpsertRequest::SetFieldOps(std::vector<FieldPartialUpdateOp>&& field_ops) {
    field_ops_ = std::move(field_ops);
}

UpsertRequest&
UpsertRequest::WithFieldOps(std::vector<FieldPartialUpdateOp>&& field_ops) {
    SetFieldOps(std::move(field_ops));
    return *this;
}

UpsertRequest&
UpsertRequest::AddFieldOp(FieldPartialUpdateOp field_op) {
    field_ops_.emplace_back(std::move(field_op));
    return *this;
}

}  // namespace milvus
