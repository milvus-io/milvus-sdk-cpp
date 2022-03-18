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

#include "milvus/types/SearchArguments.h"

#include <nlohmann/json.hpp>

namespace milvus {
struct SearchArguments::Impl {
    std::string collection_name_;
    std::set<std::string> partition_names_;
    std::set<std::string> output_field_names_;
    std::string filter_expression_;

    BinaryVecFieldDataPtr binary_vectors_;
    FloatVecFieldDataPtr float_vectors_;

    std::set<std::string> output_fields_;
    ::nlohmann::json extra_params_;

    uint64_t travel_timestamp_{0};
    uint64_t guarantee_timestamp_{GuaranteeEventuallyTs()};

    int64_t topk_{1};
    int round_decimal_{-1};

    ::milvus::MetricType metric_type_{::milvus::MetricType::L2};

    struct Validation {
        std::string param;
        int64_t min;
        int64_t max;
        bool required;

        Status
        Validate(const SearchArguments::Impl& data) const {
            auto it = data.extra_params_.find(param);
            // TODO(jibin) create a dedicate validator
            // if (it == data.extra_params_.end() && required) {
            //     return {StatusCode::INVALID_AGUMENT, "missing required parameter: " + param};
            // }
            // found, check value
            if (it != data.extra_params_.end()) {
                auto value = it.value();
                // TODO(jibin) create a dedicate validator
                // if (!value.is_number()) {
                //     return {StatusCode::INVALID_AGUMENT,
                //             "invalid value: " + param + "=" + value.dump() + ", requires number"};
                // }
                auto v = value.get<int64_t>();
                if (v < min || v > max) {
                    return {StatusCode::INVALID_AGUMENT, "invalid value: " + param + "=" + std::to_string(v) +
                                                             ", requires [" + std::to_string(min) + ", " +
                                                             std::to_string(max) + "]"};
                }
            }
            return Status::OK();
        }
    };

    Status
    Validate() const {
        auto status = Status::OK();
        auto validations = {
            Validation{"nprobe", 1, 65536, false},
            Validation{"ef", 1, 32768, false},
            Validation{"search_k", -1, 65536, false},
        };

        for (const auto& validation : validations) {
            status = validation.Validate(*this);
            if (!status.IsOk()) {
                return status;
            }
        }
        return status;
    }
};

SearchArguments::SearchArguments() : impl_(new Impl()) {
}

SearchArguments::SearchArguments(SearchArguments&&) noexcept = default;

SearchArguments::~SearchArguments() = default;

const std::string&
SearchArguments::CollectionName() const {
    return impl_->collection_name_;
}

Status
SearchArguments::SetCollectionName(const std::string& collection_name) {
    if (collection_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Collection name cannot be empty!"};
    }
    impl_->collection_name_ = collection_name;
    return Status::OK();
}

const std::set<std::string>&
SearchArguments::PartitionNames() const {
    return impl_->partition_names_;
}

Status
SearchArguments::AddPartitionName(const std::string& partition_name) {
    if (partition_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Partition name cannot be empty!"};
    }
    impl_->partition_names_.emplace(partition_name);
    return Status::OK();
}

const std::set<std::string>&
SearchArguments::OutputFields() const {
    return impl_->output_field_names_;
}

Status
SearchArguments::AddOutputField(const std::string& field_name) {
    if (field_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
    }

    impl_->output_field_names_.insert(field_name);
    return Status::OK();
}

const std::string&
SearchArguments::Expression() const {
    return impl_->filter_expression_;
}

Status
SearchArguments::SetExpression(const std::string& expression) {
    impl_->filter_expression_ = expression;
    return Status::OK();
}

FieldDataPtr
SearchArguments::TargetVectors() const {
    if (impl_->binary_vectors_ != nullptr) {
        return impl_->binary_vectors_;
    } else if (impl_->float_vectors_ != nullptr) {
        return impl_->float_vectors_;
    }

    return nullptr;
}

Status
SearchArguments::AddTargetVector(const std::string& field_name, const BinaryVecFieldData::ElementT& vector) {
    if (impl_->float_vectors_ != nullptr) {
        return {StatusCode::INVALID_AGUMENT, "Target vector must be float type!"};
    }

    if (nullptr == impl_->binary_vectors_) {
        impl_->binary_vectors_ = std::make_shared<BinaryVecFieldData>(field_name);
    }

    auto code = impl_->binary_vectors_->Add(vector);
    if (code != StatusCode::OK) {
        return {code, "Failed to add vector"};
    }

    return Status::OK();
}

Status
SearchArguments::AddTargetVector(const std::string& field_name, const FloatVecFieldData::ElementT& vector) {
    if (impl_->binary_vectors_ != nullptr) {
        return {StatusCode::INVALID_AGUMENT, "Target vector must be binary type!"};
    }

    if (nullptr == impl_->float_vectors_) {
        impl_->float_vectors_ = std::make_shared<FloatVecFieldData>(field_name);
    }

    auto code = impl_->float_vectors_->Add(vector);
    if (code != StatusCode::OK) {
        return {code, "Failed to add vector"};
    }

    return Status::OK();
}

uint64_t
SearchArguments::TravelTimestamp() const {
    return impl_->travel_timestamp_;
}

Status
SearchArguments::SetTravelTimestamp(uint64_t timestamp) {
    impl_->travel_timestamp_ = timestamp;
    return Status::OK();
}

uint64_t
SearchArguments::GuaranteeTimestamp() const {
    return impl_->guarantee_timestamp_;
}

Status
SearchArguments::SetGuaranteeTimestamp(uint64_t timestamp) {
    impl_->guarantee_timestamp_ = timestamp;
    return Status::OK();
}

Status
SearchArguments::SetTopK(int64_t topk) {
    impl_->topk_ = topk;
    return Status::OK();
}

int64_t
SearchArguments::TopK() const {
    return impl_->topk_;
}

Status
SearchArguments::SetRoundDecimal(int round_decimal) {
    impl_->round_decimal_ = round_decimal;
    return Status::OK();
}

int
SearchArguments::RoundDecimal() const {
    return impl_->round_decimal_;
}

Status
SearchArguments::SetMetricType(::milvus::MetricType metric_type) {
    impl_->metric_type_ = metric_type;
    return Status::OK();
}

::milvus::MetricType
SearchArguments::MetricType() const {
    return impl_->metric_type_;
}

Status
SearchArguments::AddExtraParam(const std::string& key, int64_t value) {
    impl_->extra_params_[key] = value;
    return Status::OK();
}

const std::string
SearchArguments::ExtraParams() const {
    return impl_->extra_params_.dump();
}

Status
SearchArguments::Validate() const {
    return impl_->Validate();
}

}  // namespace milvus
