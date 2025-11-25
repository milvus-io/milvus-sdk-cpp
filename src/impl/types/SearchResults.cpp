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

#include "milvus/types/SearchResults.h"

#include <stdexcept>

#include "../utils/Constants.h"
#include "../utils/DqlUtils.h"

namespace milvus {

SingleResult::SingleResult(const SingleResult& src)
    : pk_name_(src.pk_name_),
      score_name_(src.score_name_),
      output_fields_(src.output_fields_),
      output_names_(src.output_names_) {
    verify();
}

SingleResult::SingleResult(const std::string& pk_name, const std::string& score_name,
                           std::vector<FieldDataPtr>&& output_fields, const std::set<std::string>& output_names)
    : pk_name_(pk_name), score_name_(score_name), output_fields_(std::move(output_fields)) {
    output_names_ = output_names;
    verify();
}

void
SingleResult::verify() const {
    if (pk_name_.empty()) {
        throw std::runtime_error("Primary key name is not set");
    }
    if (score_name_.empty()) {
        throw std::runtime_error("Score field name is not set");
    }

    if (output_fields_.empty()) {
        return;
    }

    int64_t count = (output_fields_.at(0) == nullptr) ? 0 : output_fields_.at(0)->Count();
    for (const auto& field : output_fields_) {
        if (field == nullptr) {
            throw std::runtime_error("FieldData is null pointer");
        }
        if (field->Count() != count) {
            throw std::runtime_error("The lenth of output fields are unequal");
        }
    }
}

const std::vector<float>&
SingleResult::Scores() const {
    FloatFieldDataPtr score_field = OutputField<FloatFieldData>(score_name_);
    if (score_field == nullptr) {
        throw std::runtime_error("The score field data is null pointer");
    }
    return score_field->Data();
}

IDArray
SingleResult::Ids() const {
    FieldDataPtr id_field = OutputField(pk_name_);
    if (id_field == nullptr) {
        throw std::runtime_error("The primary key field data is null pointer");
    }
    if (id_field->Type() == DataType::INT64) {
        auto ptr = std::static_pointer_cast<Int64FieldData>(id_field);
        return IDArray(ptr->Data());
    } else if (id_field->Type() == DataType::VARCHAR) {
        auto ptr = std::static_pointer_cast<VarCharFieldData>(id_field);
        return IDArray(ptr->Data());
    } else {
        throw std::runtime_error("The primary key type is neither integer nor string");
    }
}

const std::string&
SingleResult::PrimaryKeyName() const {
    return pk_name_;
}

const std::string&
SingleResult::ScoreName() const {
    return score_name_;
}

const std::vector<FieldDataPtr>&
SingleResult::OutputFields() const {
    return output_fields_;
}

FieldDataPtr
SingleResult::OutputField(const std::string& name) const {
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

const std::set<std::string>&
SingleResult::OutputFieldNames() const {
    return output_names_;
}

Status
SingleResult::OutputRows(EntityRows& rows) const {
    return GetRowsFromFieldsData(output_fields_, output_names_, rows);
}

Status
SingleResult::OutputRow(int i, EntityRow& row) const {
    return GetRowFromFieldsData(output_fields_, i, output_names_, row);
}

uint64_t
SingleResult::GetRowCount() const {
    for (const auto& output_field : output_fields_) {
        if (output_field == nullptr) {
            continue;
        }
        return output_field->Count();
    }
    return 0;
}

void
SingleResult::Clear() {
    pk_name_ = "";
    score_name_ = "";
    output_fields_.clear();
    output_names_.clear();
}

/////////////////////////////////////////////////////////////////////////////////////////
SearchResults::SearchResults() = default;

SearchResults::SearchResults(std::vector<SingleResult>&& results) : nq_results_(std::move(results)) {
}

SearchResults::SearchResults(const std::vector<SingleResult>& results) : nq_results_(results) {
}

const std::vector<SingleResult>&
SearchResults::Results() const {
    return nq_results_;
}

}  // namespace milvus
