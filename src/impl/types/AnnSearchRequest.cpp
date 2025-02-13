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

#include "milvus/types/AnnSearchRequest.h"

namespace milvus {

const std::string&
AnnSearchRequest::AnnsField() const {
    return anns_field_;
}

const std::map<std::string, std::string>&
AnnSearchRequest::Param() const {
    return param_;
}

int
AnnSearchRequest::Limit() const {
    return limit_;
}

const std::string&
AnnSearchRequest::Expr() const {
    return expr_;
}

FieldDataPtr
AnnSearchRequest::TargetVectors() const {
    if (binary_vectors_ != nullptr) {
        return binary_vectors_;
    } else if (float_vectors_ != nullptr) {
        return float_vectors_;
    }

    return nullptr;
}

Status
AnnSearchRequest::AddTargetVector(std::string field_name, const std::string& vector) {
    return AddTargetVector(std::move(field_name), std::string{vector});
}

Status
AnnSearchRequest::AddTargetVector(std::string field_name, const std::vector<uint8_t>& vector) {
    return AddTargetVector(std::move(field_name), milvus::BinaryVecFieldData::CreateBinaryString(vector));
}

Status
AnnSearchRequest::AddTargetVector(std::string field_name, std::string&& vector) {
    if (float_vectors_ != nullptr) {
        return {StatusCode::INVALID_AGUMENT, "Target vector must be float type!"};
    }

    if (nullptr == binary_vectors_) {
        binary_vectors_ = std::make_shared<BinaryVecFieldData>(std::move(field_name));
    }

    auto code = binary_vectors_->Add(std::move(vector));
    if (code != StatusCode::OK) {
        return {code, "Failed to add vector"};
    }

    return Status::OK();
}

Status
AnnSearchRequest::AddTargetVector(std::string field_name, const FloatVecFieldData::ElementT& vector) {
    if (binary_vectors_ != nullptr) {
        return {StatusCode::INVALID_AGUMENT, "Target vector must be binary type!"};
    }

    if (nullptr == float_vectors_) {
        float_vectors_ = std::make_shared<FloatVecFieldData>(std::move(field_name));
    }

    auto code = float_vectors_->Add(vector);
    if (code != StatusCode::OK) {
        return {code, "Failed to add vector"};
    }

    return Status::OK();
}

Status
AnnSearchRequest::AddTargetVector(std::string field_name, FloatVecFieldData::ElementT&& vector) {
    if (binary_vectors_ != nullptr) {
        return {StatusCode::INVALID_AGUMENT, "Target vector must be binary type!"};
    }

    if (nullptr == float_vectors_) {
        float_vectors_ = std::make_shared<FloatVecFieldData>(std::move(field_name));
    }

    auto code = float_vectors_->Add(std::move(vector));
    if (code != StatusCode::OK) {
        return {code, "Failed to add vector"};
    }

    return Status::OK();
}

}  // namespace milvus
