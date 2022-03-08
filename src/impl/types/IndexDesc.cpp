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

struct IndexDesc::Impl {
    std::string field_name_;
    std::string index_name_;
    int64_t index_id_{0};
    milvus::MetricType metric_type_{milvus::MetricType::INVALID};
    milvus::IndexType index_type_{milvus::IndexType::INVALID};
    ::nlohmann::json extra_params_;

    struct Validation {
        milvus::IndexType index_type;
        std::string param;
        int64_t min;
        int64_t max;
        bool required;

        Status
        Validate(const IndexDesc::Impl& data) const {
            if (data.index_type_ == index_type) {
                auto it = data.extra_params_.find(param);
                // required and not found
                if (it == data.extra_params_.end() && required) {
                    return {StatusCode::INVALID_AGUMENT, "missing required parameter: " + param};
                }
                // found, check value
                if (it != data.extra_params_.end()) {
                    auto value = it.value();
                    if (!value.is_number()) {
                        return {StatusCode::INVALID_AGUMENT,
                                "invalid value: " + param + "=" + value.dump() + ", requires number"};
                    }
                    auto v = value.get<int64_t>();
                    if (v < min || v > max) {
                        return {StatusCode::INVALID_AGUMENT, "invalid value: " + param + "=" + std::to_string(v) +
                                                                 ", requires [" + std::to_string(min) + ", " +
                                                                 std::to_string(max) + "]"};
                    }
                }
            }
            return Status::OK();
        }
    };

    Impl() = default;

    Impl(const std::string& field_name, const std::string& index_name, int64_t index_id, milvus::MetricType metric_type,
         milvus::IndexType index_type)
        : field_name_(field_name),
          index_name_(index_name),
          index_id_(index_id),
          metric_type_(metric_type),
          index_type_(index_type) {
    }

    Status
    ValidateIndexAndMetric() const {
        if ((metric_type_ == milvus::MetricType::IP || metric_type_ == milvus::MetricType::L2) &&
            (index_type_ == milvus::IndexType::FLAT || index_type_ == milvus::IndexType::IVF_FLAT ||
             index_type_ == milvus::IndexType::IVF_SQ8 || index_type_ == milvus::IndexType::IVF_PQ ||
             index_type_ == milvus::IndexType::HNSW || index_type_ == milvus::IndexType::IVF_HNSW ||
             index_type_ == milvus::IndexType::RHNSW_FLAT || index_type_ == milvus::IndexType::RHNSW_SQ ||
             index_type_ == milvus::IndexType::RHNSW_PQ || index_type_ == milvus::IndexType::ANNOY)) {
            return Status::OK();
        }

        if ((metric_type_ == milvus::MetricType::JACCARD || metric_type_ == milvus::MetricType::TANIMOTO ||
             metric_type_ == milvus::MetricType::HAMMING) &&
            (index_type_ == milvus::IndexType::BIN_FLAT || index_type_ == milvus::IndexType::BIN_IVF_FLAT)) {
            return Status::OK();
        }

        if ((metric_type_ == milvus::MetricType::SUBSTRUCTURE || metric_type_ == milvus::MetricType::SUPERSTRUCTURE) &&
            (index_type_ == milvus::IndexType::BIN_FLAT)) {
            return Status::OK();
        }

        return {StatusCode::INVALID_AGUMENT, "Index type and metric type not match, index " +
                                                 std::to_string(index_type_) + " with metric " +
                                                 std::to_string(metric_type_)};
    }

    Status
    ValidateParams() const {
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
            status = validation.Validate(*this);
            if (!status.IsOk()) {
                return status;
            }
        }
        return status;
    }

    Status
    Validate() const {
        auto status = ValidateIndexAndMetric();
        if (status.IsOk()) {
            status = ValidateParams();
        }
        return status;
    }
};

IndexDesc::IndexDesc() : impl_(new Impl) {
}

IndexDesc::IndexDesc(const std::string& field_name, const std::string& index_name, milvus::IndexType index_type,
                     milvus::MetricType metric_type, int64_t index_id)
    : impl_{new Impl(field_name, index_name, index_id, metric_type, index_type)} {
}

IndexDesc::~IndexDesc() = default;

const std::string&
IndexDesc::FieldName() const {
    return impl_->field_name_;
}

Status
IndexDesc::SetFieldName(const std::string& field_name) {
    impl_->field_name_ = field_name;
    return Status::OK();
}

const std::string&
IndexDesc::IndexName() const {
    return impl_->index_name_;
}

Status
IndexDesc::SetIndexName(const std::string& index_name) {
    impl_->index_name_ = index_name;
    return Status::OK();
}

int64_t
IndexDesc::IndexId() const {
    return impl_->index_id_;
}

Status
IndexDesc::SetIndexId(int64_t index_id) {
    impl_->index_id_ = index_id;
    return Status::OK();
}

milvus::MetricType
IndexDesc::MetricType() const {
    return impl_->metric_type_;
}

Status
IndexDesc::SetMetricType(milvus::MetricType metric_type) {
    impl_->metric_type_ = metric_type;
    return Status::OK();
}

milvus::IndexType
IndexDesc::IndexType() const {
    return impl_->index_type_;
}

Status
IndexDesc::SetIndexType(milvus::IndexType index_type) {
    impl_->index_type_ = index_type;
    return Status::OK();
}

const std::string
IndexDesc::ExtraParams() const {
    return impl_->extra_params_.dump();
}

Status
IndexDesc::AddExtraParam(const std::string& key, int64_t value) {
    impl_->extra_params_[key] = value;
    return Status::OK();
}

Status
IndexDesc::ExtraParamsFromJson(const std::string& json) {
    try {
        impl_->extra_params_ = ::nlohmann::json::parse(json);
    } catch (const ::nlohmann::json::exception& e) {
        return {StatusCode::JSON_PARSE_ERROR, e.what()};
    }
    return Status::OK();
}

Status
IndexDesc::Validate() const {
    return impl_->Validate();
}

}  // namespace milvus
