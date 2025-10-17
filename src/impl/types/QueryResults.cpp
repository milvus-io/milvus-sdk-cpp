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

#include "milvus/types/QueryResults.h"

#include "../utils/DqlUtils.h"

namespace milvus {

QueryResults::QueryResults() = default;

QueryResults::QueryResults(const QueryResults& src)
    : output_fields_(src.output_fields_), output_names_(src.output_names_) {
}

QueryResults::QueryResults(const std::vector<FieldDataPtr>& output_fields, const std::set<std::string>& output_names) {
    output_fields_ = output_fields;
    output_names_ = output_names;
}

QueryResults::QueryResults(std::vector<FieldDataPtr>&& output_fields, const std::set<std::string>& output_names) {
    output_fields_ = std::move(output_fields);
    output_names_ = output_names;
}

FieldDataPtr
QueryResults::GetFieldByName(const std::string& name) {
    return OutputField(name);
}

FieldDataPtr
QueryResults::OutputField(const std::string& name) const {
    for (const auto& output_field : output_fields_) {
        if (output_field == nullptr) {
            continue;
        }
        if (output_field->Name() == name) {
            return output_field;
        }
    }

    return nullptr;
}

const std::vector<FieldDataPtr>&
QueryResults::OutputFields() const {
    return output_fields_;
}

const std::set<std::string>&
QueryResults::OutputFieldNames() const {
    return output_names_;
}

Status
QueryResults::OutputRows(EntityRows& rows) const {
    return GetRowsFromFieldsData(output_fields_, output_names_, rows);
}

Status
QueryResults::OutputRow(int i, EntityRow& row) const {
    return GetRowFromFieldsData(output_fields_, i, output_names_, row);
}

uint64_t
QueryResults::GetRowCount() const {
    auto data = OutputField<milvus::Int64FieldData>("count(*)");
    if (data != nullptr && data->Count() > 0) {
        return static_cast<uint64_t>(data->Value(0));
    }
    for (const auto& output_field : output_fields_) {
        if (output_field == nullptr) {
            continue;
        }
        return output_field->Count();
    }
    return 0;
}

}  // namespace milvus
