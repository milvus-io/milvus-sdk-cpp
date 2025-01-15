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
#include <string>
#include <utility>
#include <vector>

#include "milvus/Status.h"
#include "milvus/types/FieldData.h"

namespace milvus {
namespace {

struct Validation {
    std::string param;
    int64_t min;
    int64_t max;
    bool required;

    Status
    Validate(const SearchArguments&, std::unordered_map<std::string, int64_t> params) const {
        auto it = params.find(param);
        if (it != params.end()) {
            auto value = it->second;
            if (value < min || value > max) {
                return {StatusCode::INVALID_AGUMENT, "invalid value: " + param + "=" + std::to_string(value) +
                                                         ", requires [" + std::to_string(min) + ", " +
                                                         std::to_string(max) + "]"};
            }
        }
        return Status::OK();
    }
};

Status
validate(const SearchArguments& data, const std::unordered_map<std::string, int64_t>& params) {
    auto status = Status::OK();
    auto validations = {
        Validation{"nprobe", 1, 65536, false},
        Validation{"ef", 1, 32768, false},
        Validation{"search_k", -1, 65536, false},
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

const std::string&
SearchArguments::CollectionName() const {
    return collection_name_;
}

Status
SearchArguments::SetCollectionName(std::string collection_name) {
    if (collection_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Collection name cannot be empty!"};
    }
    collection_name_ = std::move(collection_name);
    return Status::OK();
}

const std::set<std::string>&
SearchArguments::PartitionNames() const {
    return partition_names_;
}

Status
SearchArguments::AddPartitionName(std::string partition_name) {
    if (partition_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Partition name cannot be empty!"};
    }
    partition_names_.emplace(std::move(partition_name));
    return Status::OK();
}

const std::set<std::string>&
SearchArguments::OutputFields() const {
    return output_field_names_;
}

Status
SearchArguments::AddOutputField(std::string field_name) {
    if (field_name.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Field name cannot be empty!"};
    }

    output_field_names_.emplace(std::move(field_name));
    return Status::OK();
}

const std::string&
SearchArguments::Expression() const {
    return filter_expression_;
}

Status
SearchArguments::SetExpression(std::string expression) {
    filter_expression_ = std::move(expression);
    return Status::OK();
}

FieldDataPtr
SearchArguments::TargetVectors() const {
    return field_data_;
}

TemplateAddTargetVectorDeclaration(, FloatVecFieldData);
TemplateAddTargetVectorDeclaration(, BinaryVecFieldData);
TemplateAddTargetVectorDeclaration(, Float16VecFieldData);
TemplateAddTargetVectorDeclaration(, BFloat16VecFieldData);

template Status
SearchArguments::AddTargetVector<BinaryVecFieldData>(std::string field_name, const std::vector<uint8_t>& vector);

template Status
SearchArguments::AddTargetVector<Float16VecFieldData>(std::string field_name, const std::vector<Eigen::half>& vector);
template Status
SearchArguments::AddTargetVector<Float16VecFieldData>(std::string field_name, const std::vector<float>& vector);
template Status
SearchArguments::AddTargetVector<Float16VecFieldData>(std::string field_name, const std::vector<double>& vector);
template Status
SearchArguments::AddTargetVector<Float16VecFieldData>(std::string field_name, const std::vector<uint8_t>& vector);

template Status
SearchArguments::AddTargetVector<BFloat16VecFieldData>(std::string field_name,
                                                       const std::vector<Eigen::bfloat16>& vector);
template Status
SearchArguments::AddTargetVector<BFloat16VecFieldData>(std::string field_name, const std::vector<float>& vector);
template Status
SearchArguments::AddTargetVector<BFloat16VecFieldData>(std::string field_name, const std::vector<double>& vector);
template Status
SearchArguments::AddTargetVector<BFloat16VecFieldData>(std::string field_name, const std::vector<uint8_t>& vector);

template <typename VecFieldDataT, typename T>
Status
SearchArguments::DoAddTargetVector(std::string field_name, const T& vector) {
    static_assert(
        std::is_same_v<FloatVecFieldData, VecFieldDataT> || std::is_same_v<BinaryVecFieldData, VecFieldDataT> ||
            std::is_same_v<Float16VecFieldData, VecFieldDataT> || std::is_same_v<BFloat16VecFieldData, VecFieldDataT>,
        "Only FloatVecFieldData, BinaryVecFieldData, Float16VecFieldData or BFloat16VecFieldData is supported");
    if (nullptr == field_data_) {
        field_data_ = std::make_shared<VecFieldDataT>(std::move(field_name));
    }

    if (auto vectors = std::dynamic_pointer_cast<VecFieldDataT>(field_data_)) {
        auto code = vectors->Add(vector);
        if (code != StatusCode::OK) {
            return {code, "Failed to add vector"};
        }
    } else {
        return {StatusCode::INVALID_AGUMENT, "Invalid vector type!"};
    }

    return Status::OK();
}

template <class VecFieldDataT, class ElementT>
Status
SearchArguments::AddTargetVector(std::string field_name, const ElementT& vector) {
    if constexpr (std::is_same_v<ElementT, std::vector<uint8_t>> &&
                  (std::is_same_v<VecFieldDataT, BinaryVecFieldData> ||
                   std::is_same_v<VecFieldDataT, Float16VecFieldData> ||
                   std::is_same_v<VecFieldDataT, BFloat16VecFieldData>)) {
        return DoAddTargetVector<VecFieldDataT>(std::move(field_name),
                                                std::move(std::string{vector.begin(), vector.end()}));
    } else {
        return DoAddTargetVector<VecFieldDataT>(std::move(field_name), vector);
    }
}

template <class VecFieldDataT>
Status
SearchArguments::AddTargetVector(std::string field_name, typename VecFieldDataT::ElementT&& vector) {
    static_assert(
        std::is_same_v<FloatVecFieldData, VecFieldDataT> || std::is_same_v<BinaryVecFieldData, VecFieldDataT> ||
            std::is_same_v<Float16VecFieldData, VecFieldDataT> || std::is_same_v<BFloat16VecFieldData, VecFieldDataT>,
        "Only FloatVecFieldData, BinaryVecFieldData, Float16VecFieldData or BFloat16VecFieldData is supported");
    if (nullptr == field_data_) {
        field_data_ = std::make_shared<VecFieldDataT>(std::move(field_name));
    }

    if (auto vectors = std::dynamic_pointer_cast<VecFieldDataT>(field_data_)) {
        auto code = vectors->Add(std::move(vector));
        if (code != StatusCode::OK) {
            return {code, "Failed to add vector"};
        }
    } else {
        return {StatusCode::INVALID_AGUMENT, "Invalid vector type!"};
    }

    return Status::OK();
}

uint64_t
SearchArguments::TravelTimestamp() const {
    return travel_timestamp_;
}

Status
SearchArguments::SetTravelTimestamp(uint64_t timestamp) {
    travel_timestamp_ = timestamp;
    return Status::OK();
}

uint64_t
SearchArguments::GuaranteeTimestamp() const {
    return guarantee_timestamp_;
}

Status
SearchArguments::SetGuaranteeTimestamp(uint64_t timestamp) {
    guarantee_timestamp_ = timestamp;
    return Status::OK();
}

Status
SearchArguments::SetTopK(int64_t topk) {
    topk_ = topk;
    return Status::OK();
}

int64_t
SearchArguments::TopK() const {
    return topk_;
}

int64_t
SearchArguments::Nprobe() const {
    if (extra_params_.find("nprobe") != extra_params_.end()) {
        return extra_params_.at("nprobe");
    }
    return 1;
}

Status
SearchArguments::SetNprobe(int64_t nprobe) {
    extra_params_["nprobe"] = nprobe;
    return Status::OK();
}

Status
SearchArguments::SetRoundDecimal(int round_decimal) {
    round_decimal_ = round_decimal;
    return Status::OK();
}

int
SearchArguments::RoundDecimal() const {
    return round_decimal_;
}

Status
SearchArguments::SetMetricType(::milvus::MetricType metric_type) {
    if (((metric_type == MetricType::IP && metric_type_ == MetricType::L2) ||
         (metric_type == MetricType::L2 && metric_type_ == MetricType::IP)) &&
        range_search_) {
        // switch radius and range_filter
        std::swap(radius_, range_filter_);
    }
    metric_type_ = metric_type;
    return Status::OK();
}

::milvus::MetricType
SearchArguments::MetricType() const {
    return metric_type_;
}

Status
SearchArguments::AddExtraParam(std::string key, int64_t value) {
    extra_params_[std::move(key)] = value;
    return Status::OK();
}

std::string
SearchArguments::ExtraParams() const {
    return ::nlohmann::json(extra_params_).dump();
}

Status
SearchArguments::Validate() const {
    return validate(*this, extra_params_);
}

float
SearchArguments::Radius() const {
    return radius_;
}

float
SearchArguments::RangeFilter() const {
    return range_filter_;
}

Status
SearchArguments::SetRange(float from, float to) {
    auto low = std::min(from, to);
    auto high = std::max(from, to);
    if (metric_type_ == MetricType::IP) {
        radius_ = low;
        range_filter_ = high;
        range_search_ = true;
    } else if (metric_type_ == MetricType::L2) {
        radius_ = high;
        range_filter_ = low;
        range_search_ = true;
    } else {
        return {StatusCode::INVALID_AGUMENT, "Metric type is not supported"};
    }
    return Status::OK();
}

bool
SearchArguments::RangeSearch() const {
    return range_search_;
}

}  // namespace milvus
