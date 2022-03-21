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

#include "milvus/types/IndexDesc.h"

#include <nlohmann/json.hpp>

#include "../TypeUtils.h"

namespace milvus {

namespace {

struct Validation {
    milvus::IndexType index_type;
    std::string param;
    int64_t min;
    int64_t max;
    bool required;

    Status
    Validate(const IndexDesc& data, std::unordered_map<std::string, int64_t> params) const {
        if (data.IndexType() == index_type) {
            auto it = params.find(param);
            // required and not found
            if (it == params.end() && required) {
                return {StatusCode::INVALID_AGUMENT, "missing required parameter: " + param};
            }
            // found, check value
            if (it != params.end()) {
                auto value = it->second;
                if (value < min || value > max) {
                    return {StatusCode::INVALID_AGUMENT, "invalid value: " + param + "=" + std::to_string(value) +
                                                             ", requires [" + std::to_string(min) + ", " +
                                                             std::to_string(max) + "]"};
                }
            }
        }
        return Status::OK();
    }
};

Status
validate_index_and_metric(const MetricType metric_type, const IndexType index_type) {
    if ((metric_type == milvus::MetricType::IP || metric_type == milvus::MetricType::L2) &&
        (index_type == milvus::IndexType::FLAT || index_type == milvus::IndexType::IVF_FLAT ||
         index_type == milvus::IndexType::IVF_SQ8 || index_type == milvus::IndexType::IVF_PQ ||
         index_type == milvus::IndexType::HNSW || index_type == milvus::IndexType::IVF_HNSW ||
         index_type == milvus::IndexType::RHNSW_FLAT || index_type == milvus::IndexType::RHNSW_SQ ||
         index_type == milvus::IndexType::RHNSW_PQ || index_type == milvus::IndexType::ANNOY)) {
        return Status::OK();
    }

    if ((metric_type == milvus::MetricType::JACCARD || metric_type == milvus::MetricType::TANIMOTO ||
         metric_type == milvus::MetricType::HAMMING) &&
        (index_type == milvus::IndexType::BIN_FLAT || index_type == milvus::IndexType::BIN_IVF_FLAT)) {
        return Status::OK();
    }

    if ((metric_type == milvus::MetricType::SUBSTRUCTURE || metric_type == milvus::MetricType::SUPERSTRUCTURE) &&
        (index_type == milvus::IndexType::BIN_FLAT)) {
        return Status::OK();
    }

    return {StatusCode::INVALID_AGUMENT, "Index type and metric type not match, index " + std::to_string(index_type) +
                                             " with metric " + std::to_string(metric_type)};
}

Status
validate_params(const IndexDesc& data, const std::unordered_map<std::string, int64_t>& params) {
    auto status = Status::OK();
    auto validations = {
        Validation{milvus::IndexType::IVF_FLAT, "nlist", 1, 65536, true},
        Validation{milvus::IndexType::IVF_SQ8, "nlist", 1, 65536, true},

        Validation{milvus::IndexType::IVF_PQ, "nlist", 1, 65536, true},
        Validation{milvus::IndexType::IVF_PQ, "m", 1, 65536, true},  // TODO: m requires mod(dim) == 0
        Validation{milvus::IndexType::IVF_PQ, "nbits", 1, 16, false},

        Validation{milvus::IndexType::HNSW, "M", 4, 64, true},
        Validation{milvus::IndexType::HNSW, "efConstruction", 8, 512, true},

        Validation{milvus::IndexType::IVF_HNSW, "nlist", 1, 65536, true},
        Validation{milvus::IndexType::IVF_HNSW, "M", 4, 64, true},
        Validation{milvus::IndexType::IVF_HNSW, "efConstruction", 8, 512, true},

        Validation{milvus::IndexType::RHNSW_FLAT, "M", 4, 64, true},
        Validation{milvus::IndexType::RHNSW_FLAT, "efConstruction", 8, 512, true},

        Validation{milvus::IndexType::RHNSW_SQ, "M", 4, 64, true},
        Validation{milvus::IndexType::RHNSW_SQ, "efConstruction", 8, 512, true},

        Validation{milvus::IndexType::RHNSW_PQ, "M", 4, 64, true},
        Validation{milvus::IndexType::RHNSW_PQ, "efConstruction", 8, 512, true},
        Validation{milvus::IndexType::RHNSW_PQ, "PQM", 1, 65536, true},  // TODO: PQM requires mod(dim) == 0

        Validation{milvus::IndexType::ANNOY, "n_trees", 1, 1024, true},
    };

    for (const auto& validation : validations) {
        status = validation.Validate(data, params);
        if (!status.IsOk()) {
            return status;
        }
    }
    return status;
}
}  // namespace

IndexDesc::IndexDesc() = default;

IndexDesc::IndexDesc(std::string field_name, std::string index_name, milvus::IndexType index_type,
                     milvus::MetricType metric_type, int64_t index_id)
    : field_name_(std::move(field_name)),
      index_name_(std::move(index_name)),
      index_type_(index_type),
      metric_type_(metric_type),
      index_id_(index_id) {
}

const std::string&
IndexDesc::FieldName() const {
    return field_name_;
}

Status
IndexDesc::SetFieldName(std::string field_name) {
    field_name_ = std::move(field_name);
    return Status::OK();
}

const std::string&
IndexDesc::IndexName() const {
    return index_name_;
}

Status
IndexDesc::SetIndexName(std::string index_name) {
    index_name_ = std::move(index_name);
    return Status::OK();
}

int64_t
IndexDesc::IndexId() const {
    return index_id_;
}

Status
IndexDesc::SetIndexId(int64_t index_id) {
    index_id_ = index_id;
    return Status::OK();
}

milvus::MetricType
IndexDesc::MetricType() const {
    return metric_type_;
}

Status
IndexDesc::SetMetricType(milvus::MetricType metric_type) {
    metric_type_ = metric_type;
    return Status::OK();
}

milvus::IndexType
IndexDesc::IndexType() const {
    return index_type_;
}

Status
IndexDesc::SetIndexType(milvus::IndexType index_type) {
    index_type_ = index_type;
    return Status::OK();
}

const std::string
IndexDesc::ExtraParams() const {
    return ::nlohmann::json(extra_params_).dump();
}

Status
IndexDesc::AddExtraParam(std::string key, int64_t value) {
    extra_params_[std::move(key)] = value;
    return Status::OK();
}

Status
IndexDesc::ExtraParamsFromJson(std::string json) {
    try {
        extra_params_ = ::nlohmann::json::parse(std::move(json));
    } catch (const ::nlohmann::json::exception& e) {
        return {StatusCode::JSON_PARSE_ERROR, e.what()};
    }
    return Status::OK();
}

Status
IndexDesc::Validate() const {
    auto status = validate_index_and_metric(metric_type_, index_type_);
    if (status.IsOk()) {
        status = validate_params(*this, extra_params_);
    }
    return status;
}

}  // namespace milvus
