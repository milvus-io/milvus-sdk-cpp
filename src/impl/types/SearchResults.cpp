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

#include "../utils/DmlUtils.h"

namespace milvus {

SingleResult::SingleResult(const std::string& pk_name, IDArray&& ids, std::vector<float>&& scores,
                           std::vector<FieldDataPtr>&& output_fields)
    : pk_name_(pk_name), ids_{std::move(ids)}, scores_{std::move(scores)}, output_fields_{std::move(output_fields)} {
}

const std::vector<float>&
SingleResult::Scores() const {
    return scores_;
}

const IDArray&
SingleResult::Ids() const {
    return ids_;
}

const std::string&
SingleResult::PrimaryKeyName() const {
    return pk_name_;
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

Status
SingleResult::setPkAndScore(int i, nlohmann::json& row) const {
    if (ids_.IsIntegerID()) {
        if (static_cast<size_t>(i) > ids_.IntIDArray().size()) {
            return Status{StatusCode::INVALID_AGUMENT, "out of bound"};
        }
        row[pk_name_] = ids_.IntIDArray().at(i);
    } else {
        if (static_cast<size_t>(i) > ids_.StrIDArray().size()) {
            return Status{StatusCode::INVALID_AGUMENT, "out of bound"};
        }
        row[pk_name_] = ids_.StrIDArray().at(i);
    }

    if (static_cast<size_t>(i) > scores_.size()) {
        return Status{StatusCode::INVALID_AGUMENT, "out of bound"};
    }
    row["score"] = scores_.at(i);
    return Status::OK();
}

Status
SingleResult::OutputRows(std::vector<nlohmann::json>& rows) const {
    auto status = GetRowsFromFieldsData(output_fields_, rows);
    if (!status.IsOk()) {
        return status;
    }

    for (auto k = 0; k < rows.size(); k++) {
        status = setPkAndScore(static_cast<int>(k), rows[k]);
        if (!status.IsOk()) {
            return status;
        }
    }
    return Status::OK();
}

Status
SingleResult::OutputRow(int i, nlohmann::json& row) const {
    auto status = GetRowFromFieldsData(output_fields_, i, row);
    if (!status.IsOk()) {
        return status;
    }

    return setPkAndScore(i, row);
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

/////////////////////////////////////////////////////////////////////////////////////////
SearchResults::SearchResults() = default;

SearchResults::SearchResults(std::vector<SingleResult>&& results) {
    nq_results_.swap(results);
}

std::vector<SingleResult>&
SearchResults::Results() {
    return nq_results_;
}

}  // namespace milvus
