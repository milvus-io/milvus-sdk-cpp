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